import datetime
import logging
import os
import sqlite3

import numpy as np
import pandas as pd

from flyhostel.data.environment import ENVIRONMENT_DB, MAX_GAP_S, load_readings, load_events, interpolate
from flyhostel.utils.utils import get_frame_time_origin

logger = logging.getLogger(__name__)


class EnvironmentLoader:
    """
    Temperature (C) and relative humidity (%) of the incubator, from ENVIRONMENT_DB
    (see flyhostel.data.environment)

    loader.load_environment()                  # self.environment: readings during the experiment, with t (ZT s)
    loader.annotate_environment(loader.dt)     # adds temperature and humidity to a dataframe with t or frame_number
    loader.environment_at_frame(frame_number)  # {"temperature": ..., "humidity": ...}
    """
    basedir=None
    dbfile=None
    meta_info=None
    environment=None
    environment_events=None

    def _environment_origin(self):
        """unix s (UTC) of frame_time 0"""
        origin=get_frame_time_origin(self.basedir, dbfile=self.dbfile)
        return origin.replace(tzinfo=datetime.timezone.utc).timestamp()

    def _frame_time_range(self):
        with sqlite3.connect(os.path.join(self.basedir, "index.db")) as conn:
            return conn.execute("SELECT MIN(frame_time), MAX(frame_time) FROM frames").fetchone()

    def load_environment(self, db=ENVIRONMENT_DB):
        origin=self._environment_origin()
        first, last=self._frame_time_range()
        start, end=origin + first/1000, origin + last/1000
        self.environment=load_readings(start, end, db=db)
        self.environment_events=load_events(start, end, db=db)
        for df in [self.environment, self.environment_events]:
            df["t"]=df["timestamp"] - origin + self.meta_info["t_after_ref"]
        if len(self.environment)==0:
            logger.warning("No incubator readings for %s", self.basedir)
        return self.environment

    def _frame_times(self, frame_numbers):
        """frame_time (s) of each frame number, from the imgstore index"""
        frame_numbers=np.asarray(frame_numbers, dtype=np.int64)
        with sqlite3.connect(os.path.join(self.basedir, "index.db")) as conn:
            lo, hi=int(frame_numbers.min()), int(frame_numbers.max())
            index=pd.read_sql(
                "SELECT frame_number, frame_time FROM frames WHERE frame_number BETWEEN ? AND ?",
                conn, params=(lo, hi)
            ).set_index("frame_number")["frame_time"]
        return index.reindex(frame_numbers).values/1000

    def annotate_environment(self, df, max_gap=MAX_GAP_S):
        """
        Add temperature and humidity to df, by t (ZT s) if it has it, else by frame_number
        NaN where the incubator has no readings (less than max_gap s apart) around that time
        """
        if self.environment is None:
            self.load_environment()
        origin=self._environment_origin()

        if "t" in df.columns:
            frame_time=df["t"].values - self.meta_info["t_after_ref"]
        else:
            frame_time=self._frame_times(df["frame_number"].values)

        values=interpolate(self.environment, origin + frame_time, max_gap=max_gap)
        df["temperature"]=values["temperature"]
        df["humidity"]=values["humidity"]
        return df

    def environment_at_frame(self, frame_number):
        df=self.annotate_environment(pd.DataFrame({"frame_number": [frame_number]}))
        return df[["temperature", "humidity"]].iloc[0].to_dict()
