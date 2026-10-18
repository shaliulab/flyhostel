"""
Temperature and humidity of the incubator the FlyHostels live in

The incubator (RUMED Control2015.touch) logs one reading per minute and serves one log per day
(sometimes more, e.g. after a power failure) at INCUBATOR_URL. `sync` copies them into ENVIRONMENT_DB:

* readings (incubator, timestamp, temperature, humidity, co2): timestamp = unix seconds, UTC
* events (incubator, timestamp, type, status, message): door opened / closed, alarms, startup...
* logs (incubator, name, date, readings, events, downloaded): logs already imported

The incubator's clock is UTC (no DST changes) and is not synchronized: it runs some minutes
ahead (+7 min in Oct 2026). Each sync stores the offset it measured (server Date header - local UTC)
in `clock`, it is not applied to the timestamps.

    python -m flyhostel.data.environment sync          # new logs (and the last day again)
    python -m flyhostel.data.environment sync --all    # every log the incubator has
    python -m flyhostel.data.environment show FlyHostel1_4X_2026-10-13_17-00-00
"""

import argparse
import datetime
import email.utils
import logging
import os
import re
import sqlite3
import urllib.request

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ENVIRONMENT_DB = os.environ.get("FLYHOSTEL_ENVIRONMENT", "/flyhostel_data/environment/environment.db")
INCUBATOR_URL = os.environ.get("INCUBATOR_URL", "http://10.43.205.5:30000")
# Readings further apart than this are not interpolated (incubator off, logger down)
MAX_GAP_S = 300
TIMEOUT_S = 30

LOG_RE = re.compile(r'csv\.php\?Log=(log/\d+/(\d+-(\d{8})-\d+)\.log)')

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    incubator TEXT NOT NULL, timestamp INTEGER NOT NULL,
    temperature REAL, humidity REAL, co2 REAL,
    PRIMARY KEY (incubator, timestamp)
);
CREATE TABLE IF NOT EXISTS events (
    incubator TEXT NOT NULL, timestamp INTEGER NOT NULL,
    type TEXT, status TEXT, message TEXT,
    PRIMARY KEY (incubator, timestamp, type, message)
);
CREATE TABLE IF NOT EXISTS logs (
    incubator TEXT NOT NULL, name TEXT NOT NULL, date TEXT,
    readings INTEGER, events INTEGER, downloaded TEXT,
    PRIMARY KEY (incubator, name)
);
CREATE TABLE IF NOT EXISTS clock (
    incubator TEXT NOT NULL, measured TEXT NOT NULL, offset_s REAL
);
"""


def connect(db=ENVIRONMENT_DB, create=False):
    if create:
        os.makedirs(os.path.dirname(db), exist_ok=True)
    elif not os.path.exists(db):
        raise FileNotFoundError(f"{db} does not exist. Run python -m flyhostel.data.environment sync")
    conn = sqlite3.connect(db, timeout=60)
    if create:
        conn.executescript(SCHEMA)
    return conn


def _get(url):
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:
        return response.read().decode("utf-8", errors="replace"), response.headers.get("Date")


def list_logs(url=INCUBATOR_URL):
    """[(path, name, YYYYMMDD)] of every log of the incubator, oldest first; and its clock offset (s)"""
    html, date_header = _get(f"{url}/index.php")
    logs = sorted(set(LOG_RE.findall(html)), key=lambda log: log[1])
    offset = None
    if date_header:
        server_time = email.utils.parsedate_to_datetime(date_header)
        offset = (server_time - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
    return logs, offset


def _timestamp(date, time):
    dt = datetime.datetime.strptime(f"{date} {time}", "%d.%m.%Y %H:%M:%S")
    return int(dt.replace(tzinfo=datetime.timezone.utc).timestamp())


def _number(value):
    value = value.strip().strip('"').replace(",", ".")
    return float(value) if value else None


def parse_log(text):
    """Serial number, readings [(timestamp, temperature, humidity, co2)] and events
    [(timestamp, type, status, message)] of the csv export of one log"""
    serial = None
    readings, events = [], []
    section = None
    for line in text.splitlines():
        fields = line.rstrip("\r").split("\t")
        if fields[0] == '"SN#:"':
            serial = fields[1].strip('"')
        elif fields[0] == '"Datum"':
            section = "events" if '"Typ"' in fields else "readings"
        elif section and re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", fields[0]):
            fields += [""] * (5 - len(fields))
            timestamp = _timestamp(fields[0], fields[1])
            if section == "readings":
                readings.append((timestamp, *(_number(value) for value in fields[2:5])))
            else:
                events.append((timestamp, *(value.strip() for value in fields[2:5])))
    return serial, readings, events


def sync(url=INCUBATOR_URL, db=ENVIRONMENT_DB, everything=False):
    """Import the logs not imported yet (all with everything=True). The logs of the
    last day imported are imported again, the incubator keeps writing them."""
    logs, offset = list_logs(url)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    with connect(db, create=True) as conn:
        done = {name for name, in conn.execute("SELECT name FROM logs")}
        last_date = conn.execute("SELECT MAX(date) FROM logs").fetchone()[0]
        if not everything:
            logs = [log for log in logs if log[1] not in done or log[2] >= last_date]

        for path, name, date in logs:
            text, _ = _get(f"{url}/csv.php?Log={path}")
            serial, readings, events = parse_log(text)
            if serial is None:
                logger.warning("%s: no serial number, skipped", name)
                continue
            conn.executemany(
                "INSERT OR REPLACE INTO readings VALUES (?, ?, ?, ?, ?)",
                [(serial, *reading) for reading in readings]
            )
            conn.executemany(
                "INSERT OR REPLACE INTO events VALUES (?, ?, ?, ?, ?)",
                [(serial, *event) for event in events]
            )
            conn.execute(
                "INSERT OR REPLACE INTO logs VALUES (?, ?, ?, ?, ?, ?)",
                (serial, name, date, len(readings), len(events), now)
            )
            conn.commit()
            logger.info("%s: %s readings, %s events", name, len(readings), len(events))

        if offset is not None and logs:
            conn.execute("INSERT INTO clock VALUES (?, ?, ?)", (serial, now, offset))
    return len(logs)


def load_readings(start, end, db=ENVIRONMENT_DB, incubator=None, margin=MAX_GAP_S):
    """Readings between start and end (datetime in UTC, naive or aware, or unix seconds)"""
    start, end = (_to_unix(value) for value in (start, end))
    query = "SELECT * FROM readings WHERE timestamp BETWEEN ? AND ?"
    params = [start - margin, end + margin]
    if incubator is not None:
        query += " AND incubator = ?"
        params.append(incubator)
    with connect(db) as conn:
        readings = pd.read_sql(query + " ORDER BY timestamp", conn, params=params)
    if readings["incubator"].nunique() > 1:
        raise ValueError(f"Readings of several incubators ({readings['incubator'].unique()}), pass incubator=")
    return readings


def load_events(start, end, db=ENVIRONMENT_DB):
    with connect(db) as conn:
        return pd.read_sql(
            "SELECT * FROM events WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp", conn,
            params=[_to_unix(start), _to_unix(end)]
        )


def interpolate(readings, timestamps, max_gap=MAX_GAP_S):
    """temperature and humidity at each timestamp (unix s), linearly interpolated between
    the readings around it; NaN where those are more than max_gap s apart or missing"""
    timestamps = np.asarray(timestamps, dtype=float)
    out = {}
    t = readings["timestamp"].values.astype(float)
    if len(t) == 0:
        return {column: np.full(len(timestamps), np.nan) for column in ("temperature", "humidity")}

    after = np.clip(np.searchsorted(t, timestamps), 1, len(t) - 1) if len(t) > 1 else np.zeros(len(timestamps), int)
    before = np.maximum(after - 1, 0)
    valid = (t[before] <= timestamps) & (timestamps <= t[after]) & (t[after] - t[before] <= max_gap)
    # exactly on a reading
    valid |= np.isin(timestamps, t)
    for column in ("temperature", "humidity"):
        values = np.interp(timestamps, t, readings[column].values.astype(float))
        values[~valid] = np.nan
        out[column] = values
    return out


def _to_unix(value):
    if isinstance(value, datetime.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=datetime.timezone.utc)
        return value.timestamp()
    return float(value)


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=ENVIRONMENT_DB)
    subparsers = parser.add_subparsers(dest="command", required=True)
    sync_parser = subparsers.add_parser("sync", help="Download the incubator logs into the database")
    sync_parser.add_argument("--url", default=INCUBATOR_URL)
    sync_parser.add_argument("--all", action="store_true", help="Download every log again")
    show_parser = subparsers.add_parser("show", help="Summary of the environment during an experiment")
    show_parser.add_argument("experiment", help="FlyHostelN_NX_YYYY-MM-DD_HH-MM-SS, or its folder")
    return parser


def main(args=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = get_parser().parse_args(args)
    if args.command == "sync":
        n = sync(url=args.url, db=args.db, everything=args.all)
        print(f"{n} logs imported into {args.db}")
    elif args.command == "show":
        from flyhostel.utils import get_basedir, get_experiment_start
        basedir = args.experiment if os.path.isdir(args.experiment) else get_basedir(args.experiment)
        start = get_experiment_start(basedir)
        index = sqlite3.connect(os.path.join(basedir, "index.db"))
        last = index.execute("SELECT MAX(frame_time) FROM frames").fetchone()[0]
        end = start + datetime.timedelta(milliseconds=last)
        readings = load_readings(start, end, db=args.db)
        print(f"{basedir}: {start} - {end} UTC, {len(readings)} readings")
        print(readings[["temperature", "humidity"]].describe().round(2).to_string())
        print(load_events(start, end, db=args.db).to_string())


if __name__ == "__main__":
    main()
