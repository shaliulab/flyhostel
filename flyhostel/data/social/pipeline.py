"""
Run the social pipelines on a collection of FlyHostelLoaders.

    run(loaders, pose=False)   PIPELINE 1  centroid proximity -> approach attribution
                                           -> classification -> approach/response (centroid grade)
    run(loaders, pose=True)    PIPELINE 2  + pose contacts (inside centroid gate windows)
                                           -> contact classification -> approach/response (pose grade)

Loaders are grouped by experiment (only flies sharing an arena can interact). Experiments
run in parallel processes (n_jobs); inside one experiment, centroids load in parallel over
flies and pose contacts in parallel over time blocks (n_jobs_fly). The GPU (cupy) only
accelerates the distance kernels; the dominant costs are SQLite and .h5 decompression,
which is why parallelism is over processes, not the GPU.

Outputs (one feather per table, prefixed by the experiment):
    proximity   centroid bouts, oriented and classified
    summary     per fly and hour: bouts, time interacting, bouts initiated
    sleep       sleep bouts used for latencies
    contacts    pose contact bouts, oriented and classified            (pose=True)
    approach    approach->response events with grade and sleep latency
    passby      an active fly enters and leaves a radius around a focal fly, with the
                sleep/wake state durations of both flies around the event
"""
import argparse
import json
import logging
import os
from collections import defaultdict

import joblib

from .approach_response import approach_response
from .centroid import annotate_approach, interaction_summary, orient, proximity_bouts
from .classify import classify_bouts
from .config import SocialConfig
from .contact import NODES, detect_contacts
from .movement import asleep_from_loaders, compute_movement
from .passby import FOCAL_STATES, passby_events
from .tracks import load_group_tracks, make_pose_readers

logger = logging.getLogger(__name__)


def _orient_and_classify(bouts, tracks, movement, cfg):
    bouts = annotate_approach(bouts, tracks, movement, cfg.proximity.approach_s)
    return classify_bouts(orient(bouts), cfg.movement, cfg.response)


def run_experiment(loaders, cfg=None, pose=True, sleep_source="centroid", min_time=None, max_time=None,
                   use_gpu=False, n_jobs_fly=1, output=None):
    """Both pipelines for the flies of ONE experiment. Returns a dict of DataFrames.

    Arguments:
        sleep_source (str): "centroid" = immobility >= cfg.movement.min_sleep_s on the
            centroid tracks; "loader" = loader.load_sleep_data (behaviour-based)
        output (str): if given, tables are written to {output}/{experiment}_{table}.feather
    """
    cfg = cfg or SocialConfig()
    tracks = load_group_tracks(loaders, cfg.tracks, min_time=min_time, max_time=max_time, n_jobs=n_jobs_fly)
    asleep = asleep_from_loaders(loaders, tracks, cfg.movement.min_sleep_s) if sleep_source == "loader" else None
    movement = compute_movement(tracks, cfg.movement, asleep=asleep)
    readers = make_pose_readers(loaders, tracks, cfg.tracks, cfg.contact, nodes=NODES) if pose else None

    out = analyze(tracks, movement, cfg, readers=readers, use_gpu=use_gpu, n_jobs=n_jobs_fly)
    if output is not None:
        write_tables(out, tracks.experiment, output, cfg)
    return out


def analyze(tracks, movement, cfg, readers=None, use_gpu=False, n_jobs=1):
    """Everything after loading. readers=None runs the centroid pipeline only."""
    p = cfg.proximity
    prox = proximity_bouts(tracks, p.max_dist_mm, p.min_duration_s, p.merge_gap_s, use_gpu=use_gpu)
    prox = _orient_and_classify(prox, tracks, movement, cfg)
    sleep = movement.sleep_bouts.assign(
        experiment=tracks.experiment,
        id=lambda df: [tracks.ids[f] for f in df["fly"]],
        t_start=lambda df: tracks.t[df["start_idx"]],
    )
    out = {
        "proximity": prox,
        "summary": interaction_summary(prox, tracks),
        "sleep": sleep,
        "passby": passby_events(tracks, movement, cfg.passby, cfg.movement.move_speed_mm_s,
                                p.approach_s, use_gpu=use_gpu),
    }

    if readers is not None:
        c = cfg.contact
        candidates = proximity_bouts(tracks, c.gate_mm, 0, c.merge_gap_s, use_gpu=use_gpu)
        contacts = detect_contacts(tracks, readers, candidates, c, use_gpu=use_gpu, n_jobs=n_jobs)
        if not contacts.empty:
            contacts = _orient_and_classify(contacts, tracks, movement, cfg)
        out["contacts"] = contacts
        events_from, all_bouts = contacts, contacts
    else:
        events_from, all_bouts = prox, prox

    out["approach"] = (
        approach_response(events_from, tracks, movement, all_bouts, cfg, readers=readers)
        if len(events_from) else events_from.iloc[0:0]
    )
    return out


def write_tables(tables, experiment, output, cfg):
    os.makedirs(output, exist_ok=True)
    for name, df in tables.items():
        path = os.path.join(output, f"{experiment}_{name}.feather")
        df.reset_index(drop=True).to_feather(path)
        logger.info("%s: %d rows -> %s", name, len(df), path)
    with open(os.path.join(output, f"{experiment}_social_config.json"), "w") as handle:
        json.dump(cfg.to_dict(), handle, indent=2)


def run(loaders, n_jobs=1, **kwargs):
    """``run_experiment`` for every experiment represented in ``loaders``, in parallel.

    Returns {experiment: {table: DataFrame}}.
    """
    groups = defaultdict(list)
    for loader in loaders:
        groups[loader.experiment].append(loader)
    if kwargs.get("use_gpu") and n_jobs != 1:
        logger.warning("use_gpu with n_jobs=%s: every process will share the same GPU", n_jobs)
    results = joblib.Parallel(n_jobs=min(n_jobs, len(groups)))(
        joblib.delayed(run_experiment)(group, **kwargs) for group in groups.values()
    )
    return dict(zip(groups.keys(), results))


def get_parser():
    ap = argparse.ArgumentParser(description="Social interactions of a FlyHostel experiment")
    ap.add_argument("--experiment", nargs="+", required=True)
    ap.add_argument("--pose", action="store_true", help="Run the pose pipeline too")
    ap.add_argument("--output", default=".")
    ap.add_argument("--fps", type=float, default=SocialConfig().tracks.fps)
    ap.add_argument("--min-time", type=float, default=None, help="ZT seconds")
    ap.add_argument("--max-time", type=float, default=None, help="ZT seconds")
    ap.add_argument("--sleep-source", choices=["centroid", "loader"], default="centroid")
    ap.add_argument("--n-jobs", type=int, default=1, help="experiments in parallel")
    ap.add_argument("--n-jobs-fly", type=int, default=1, help="flies / time blocks in parallel")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--passby-radius", type=float, default=SocialConfig().passby.radius_mm, help="mm")
    ap.add_argument("--focal-state", choices=FOCAL_STATES, default=SocialConfig().passby.focal_state,
                    help="state of the focal fly in pass-by events; 'awake' = negative control")
    return ap


def main(args=None):
    from flyhostel.data.pose.main import init_loaders

    args = get_parser().parse_args(args)
    cfg = SocialConfig()
    cfg.tracks.fps = args.fps
    cfg.passby.radius_mm = args.passby_radius
    cfg.passby.focal_state = args.focal_state
    loaders = [loader for experiment in args.experiment for loader in init_loaders(experiment, metadata={})]
    run(
        loaders, n_jobs=args.n_jobs, cfg=cfg, pose=args.pose, sleep_source=args.sleep_source,
        min_time=args.min_time, max_time=args.max_time, use_gpu=args.gpu,
        n_jobs_fly=args.n_jobs_fly, output=args.output,
    )
