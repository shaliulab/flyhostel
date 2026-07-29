"""
proboscis_candidates.py   —   NORMAL environment (numpy / h5py / flyhostel)

Classify every raw proboscis detection against the production pipeline's decision,
and emit review records split into tiers. The SLEAP-env builder (build_review_slp.py)
turns these records into .slp review projects.

The whole design rests on ONE idea:

    A detection is judged on two INDEPENDENT axes —
        AXIS 1  pipeline decision : does production KEEP it?   pc >= PROD_THRESH
        AXIS 2  our geometry      : is it a PLAUSIBLE real extension?  (shared gate)
    The geometry gate is computed ONCE and applied IDENTICALLY to every detection.
    A tier is therefore just  plausible × (side of PROD_THRESH)  — nothing else.
    That is what makes the 'likely' tier a true POSITIVE CONTROL of 'potential':
    they share the exact same processing and differ only in confidence.

Runs on RAW predictions (before the production gpu_filters pipeline), because that
filter imputes proboscis->head, RLE-flattens, and smooths — all of which destroy
the extension signal.
"""
import os
import logging
import numpy as np
import pandas as pd
import h5py
import yaml
import joblib
from tqdm.auto import tqdm

from flyhostel.utils import (
    get_framerate, get_pixels_per_mm, get_chunksize,
    get_basedir, get_dbfile, get_number_of_animals, get_local_identities,
)

from flyhostel.utils.pose_export import (
    load_arrays, check_file_contains_everything_needed, get_first_frame_number, estimate_body_length_mm
)
from flyhostel.data.pose.main import FlyHostelLoader
N_JOBS = -1

logger = logging.getLogger(__name__)

# ==========================================================================
# ||||||||||||||||||  MANUAL PARAMETERS (not data-derived)  ||||||||||||||||||
# ||  Everything below is a human choice. Data-derived gates (max_ext_mm,   ||
# ||  cone_rad, max_tip_vel_mm_s) are fitted by derive_parameters and       ||
# ||  cached to params.yaml — they are NOT here.                            ||
# ==========================================================================

# --- the decision we are validating ---
PROD_THRESH   = 0.5   # proboscis confidence production requires to KEEP a detection.
                       #   unit: SLEAP peak score (unitless; this model's scores run ~0..1.4).
                       #   >= PROD_THRESH  => pipeline keeps  (AXIS 1 = True)

# --- review floor ---
CONF_LOW      = 0.30   # below this a sub-threshold detection is background, not a candidate FN.
                       #   unit: SLEAP peak score (unitless).

# --- what counts as an "extension" (vs a retracted proboscis) ---
EXT_FRAC      = 0.30   # a detection is "extended" iff head->proboscis distance
                       #   > EXT_FRAC * max_ext_px.  unit: fraction of physiological max reach.

# --- axis-trust / held-dropout thresholds ---
CONF_HI_FIT  = 1.00   # strict: only top-quality detections fit the geometry gates
                      #   (keep high so max_ext_mm etc. aren't inflated by marginal detections)
AXIS_CONF_MIN = 0.75  # separate, lower bar for "body axis is trustworthy enough to
                      #   ENFORCE the cone" (thorax ~0.85 is a perfectly good axis)



HEAD_CONF_MIN = 0.80   # head confidence required to claim "proboscis missing but should be
                       #   visible" (held-dropout FN).  unit: SLEAP score (unitless).

# --- parameter fitting (derive_parameters) ---
SUBSAMPLE_PER_FILE = 300_000   # cap of confident values pooled per file for percentiles.
                               #   unit: count (memory bound; not physical).
EXT_P,  EXT_PAD  = 99.9, 1.25  # reach ceiling  = EXT_PAD * p(EXT_P) of head->prob distance (mm).
CONE_P, CONE_PAD = 99.5, 1.30  # cone half-width = CONE_PAD * p(CONE_P) of off-axis angle (rad).
VEL_P,  VEL_PAD  = 99.9, 1.50  # tip-velocity ceiling = VEL_PAD * p(VEL_P) of tip speed (mm/s).

# --- IO ---
OUTPUT_DIR  = "proboscis_qc"   # params.yaml + per-fly candidate CSVs
RECORDS_DIR = "records"        # per-(fly,tier) feathers consumed by the SLEAP builder

# tiers we emit records for. TN (dropped & geometry-invalid) is the huge correctly-
# rejected background and is NOT emitted.
TIER_POTENTIAL = "potential"           # candidate FALSE NEGATIVES  (the review set)
TIER_LIKELY    = "likely"              # candidate TRUE POSITIVES    (positive control)
TIER_FP        = "confident_rejected"  # candidate FALSE POSITIVES   (confident but geometry-invalid,
                                       #                              e.g. proboscis-on-abdomen)
RNG = np.random.default_rng(0)


# header param
INST_SCORE_MIN = 8      # whole-instance (skeleton) confidence floor. Below this the pose
                        #   is unreliable as a whole (awkward posture / flight), even if the
                        #   proboscis node peak is high.  unit: SLEAP instance score (unitless).

PARAMS_FILE = "./params.yaml"

PARAMS={
    "cone_rad": 0.697681725025177,
    "max_ext_mm": 1.1090800762176514,
    "max_tip_vel_mm_s": 19.673828125,
}



def compute_geometry(locs, sc, nodes, inst_scores, bp1="head", bp2="thorax"):
    """As before, plus body long-axis heading, body length, and abdomen confidence.
 
    New keys
    --------
    body_theta   : (frames, tracks) heading (rad) of the long axis, abdomen->head
                   (anterior). This is the axis meant by MAX_ANGLE in the detector.
    body_length  : (frames, tracks) head<->abdomen distance in PIXELS.
    abd_conf     : (frames, tracks) abdomen point confidence.
 
    Note: bp2 default corrected to "thorax" (was a typo "throax"); pass bp2
    explicitly if your node graph names it differently.
    """
    pi, hi, ti = nodes.index("proboscis"), nodes.index(bp1), nodes.index(bp2)
    ab = nodes.index("abdomen")
    prob_conf, head_conf, axis_conf = sc[:, pi, :], sc[:, hi, :], sc[:, ti, :]
    abd_conf = sc[:, ab, :]                                     # NEW
 
    body_axis = locs[:, hi, :, :] - locs[:, ti, :, :]          # thorax -> head (anterior)
    theta = np.arctan2(body_axis[:, 1, :], body_axis[:, 0, :])
    vec = locs[:, pi, :, :] - locs[:, hi, :, :]                # head -> proboscis
    dist = np.linalg.norm(vec, axis=1)
    ang = np.arctan2(vec[:, 1, :], vec[:, 0, :])
    off_axis = np.abs(np.angle(np.exp(1j * (ang - theta))))
 
    along = vec[:, 0, :] * np.cos(theta) + vec[:, 1, :] * np.sin(theta)
    prob_to_abd = np.linalg.norm(locs[:, pi, :, :] - locs[:, ab, :, :], axis=1)
 
    # NEW: long axis (abdomen -> head, anterior) and body length (head <-> abdomen)
    long_axis = locs[:, hi, :, :] - locs[:, ab, :, :]
    body_theta = np.arctan2(long_axis[:, 1, :], long_axis[:, 0, :])
    body_length = np.linalg.norm(long_axis, axis=1)            # px
 
    disp = np.full_like(dist, np.nan)
    disp[1:, :] = np.linalg.norm(np.diff(locs[:, pi, :, :], axis=0), axis=1)
 
    return dict(pi=pi, hi=hi, ti=ti, ab=ab,
                prob_conf=prob_conf, head_conf=head_conf, axis_conf=axis_conf,
                abd_conf=abd_conf,
                dist=dist, off_axis=off_axis, along=along, prob_to_abd=prob_to_abd,
                theta=theta, body_theta=body_theta, body_length=body_length,
                disp=disp, locs=locs, inst_scores=inst_scores)
 


# --------------------------------------------------------------------------- #
# Bridge: pose file -> detector columns on loader.dt                           #
# --------------------------------------------------------------------------- #
def attach_pose_features(
    loader, *, track, first_frame_number,
    orientation_col="heading", body_length_col="body_length_mm",
    body_length_estimator="percentile", body_length_percentile=90.0,
    min_pose_confidence=0.0, pose_kind="raw", inplace=True,
):
    """Populate ``loader.dt`` with heading (rad) and body length (mm) per frame.
 
    Parameters that only *you* can resolve (they depend on the FlyHostel layout,
    not on anything visible here) are required explicitly:
      * ``track``              : index of this fly's track in its pose file.
      * ``first_frame_number`` : frame_number of pose-array index 0, e.g.
                                 ``get_first_frame_number(path, get_chunksize(exp))``.
 
    Alignment: pose array row ``k`` -> ``frame_number = first_frame_number + k``.
    The result is left-joined onto ``loader.dt`` by ``frame_number``, so rows with
    no pose stay NaN (and the detector treats them as "can't assert closeness").
 
    Returns the augmented dt (and assigns it to ``loader.dt`` if ``inplace``).
    """

    raise Exception # this function is a fossile?

    if h5py is None:
        raise RuntimeError("h5py is required for attach_pose_features but is unavailable.")
 
    path = loader.get_pose_file_h5py(pose_kind)
    locs, sc, nodes, inst = load_arrays(path)
    g = compute_geometry(locs, sc, nodes, inst)
 
    heading = g["body_theta"][:, track]                    # long-axis anterior heading (rad)
    conf = np.minimum(g["head_conf"][:, track], g["abd_conf"][:, track])
    body_len_mm = estimate_body_length_mm(
        g["body_length"][:, track], conf, loader.pixels_per_mm,
        estimator=body_length_estimator, percentile=body_length_percentile,
        min_confidence=min_pose_confidence,
    )
 
    n = heading.shape[0]
    fn = np.arange(n, dtype=np.int64) + int(first_frame_number)
    pose_df = pd.DataFrame({
        "frame_number": fn,
        orientation_col: heading,
        body_length_col: body_len_mm,
    })
 
    dt = loader.dt.merge(pose_df, on="frame_number", how="left")
    if inplace:
        loader.dt = dt
    return dt

# ==========================================================================
# parameter fitting (run once, cached). Physiological units -> rig-invariant.
# ==========================================================================
def _subsample(x):
    x = x[np.isfinite(x)]
    if x.size > SUBSAMPLE_PER_FILE:
        x = RNG.choice(x, SUBSAMPLE_PER_FILE, replace=False)
    return x




# ==========================================================================
# ||||||||||||||||||||||||  CLASSIFICATION CORE  ||||||||||||||||||||||||||||
# ==========================================================================
def gate_masks(locs, g, params, ppm, fps, track=0):
    """THE single definition of the shared geometry gate.

    Returns (gates, plausible, vals) where `gates` is an ordered dict of per-frame
    boolean arrays and `plausible` is their AND. classify_detections AND explain_frame
    both call this, so the explanation can never drift from the classification.
    """
    from collections import OrderedDict

    max_ext_px = params["max_ext_mm"] * ppm
    cone_rad   = params["cone_rad"]
    jump_px    = params["max_tip_vel_mm_s"] / fps * ppm
    ext_min_px = EXT_FRAC * max_ext_px

    dist = g["dist"][:, track];  off  = g["off_axis"][:, track]
    pc   = g["prob_conf"][:, track]; hc = g["head_conf"][:, track]
    ac   = g["axis_conf"][:, track]; disp = g["disp"][:, track]
    along = g["along"][:, track];    isc = g["inst_scores"][:, track]

    detected = np.isfinite(dist)
    axis_ok  = ac > AXIS_CONF_MIN

    d_out = np.full_like(dist, np.nan); d_out[:-1] = disp[1:]
    d_skip = np.full_like(dist, np.nan)
    d_skip[1:-1] = np.linalg.norm(
        locs[2:, g["pi"], :, track] - locs[:-2, g["pi"], :, track], axis=1)
    excursion = (disp > jump_px) & (d_out > jump_px) & (d_skip < jump_px)

    gates = OrderedDict([
        ("detected",      detected),
        ("in_reach",      dist <= max_ext_px),
        ("in_cone",       (~axis_ok) | (off <= cone_rad)),
        ("anterior",      along > 0),
        ("extended",      dist > ext_min_px),
        ("not_excursion", ~excursion),
        ("inst_ok",       isc >= INST_SCORE_MIN),
    ])
    plausible = detected.copy()
    for m in gates.values():
        plausible = plausible & m

    vals = dict(dist=dist, off=off, pc=pc, hc=hc, ac=ac, along=along, isc=isc,
                axis_ok=axis_ok, excursion=excursion,
                max_ext_px=max_ext_px, cone_rad=cone_rad, ext_min_px=ext_min_px,
                jump_px=jump_px, ppm=ppm)
    return gates, plausible, vals


def tier_masks(pipeline_keeps, plausible, pc):
    """THE single definition of the TP/FP/FN truth table. Works elementwise on arrays
    (classify_detections) and on scalars (explain_frame).

    NB: coerced to np.bool_ because Python's `~True` is -2, not False -- a scalar call
    would silently produce garbage otherwise.
    """
    keeps = np.asarray(pipeline_keeps, dtype=bool)
    plaus = np.asarray(plausible, dtype=bool)
    is_TP = keeps & plaus
    is_FP = keeps & ~plaus
    is_FN = (~keeps) & plaus & (np.asarray(pc) >= CONF_LOW)
    return is_TP, is_FP, is_FN


def classify_detections(path, params, track=0):
    """Return a per-frame DataFrame of candidates, each tagged with classification,
    tier and reason. The geometry gate below is the SINGLE point where plausibility
    is decided — it is identical for every detection, so tiers cannot diverge."""
    locs, sc, nodes, inst_scores = load_arrays(path)
    g = compute_geometry(locs, sc, nodes, inst_scores)
    exp = os.path.basename(path).split("__")[0]
    ppm, fps = get_pixels_per_mm(exp), get_framerate(exp)

    # ---------------------------------------------------------------------
    # SHARED GEOMETRY GATE  (the ONE definition; explain_frame calls it too)
    # ---------------------------------------------------------------------
    gates, plausible, v = gate_masks(locs, g, params, ppm, fps, track)
    detected = gates["detected"]
    dist, off, pc, hc = v["dist"], v["off"], v["pc"], v["hc"]
    prob_to_abd = g["prob_to_abd"][:, track]
    extended = gates["extended"]

    # ---------------------------------------------------------------------
    # ####################  TP / FP / FN / TN  ############################
    #   AXIS 1  pipeline_keeps = detected & (pc >= PROD_THRESH)
    #   AXIS 2  plausible      = shared geometry gate above
    #
    #   pipeline_keeps  plausible -> classification   tier                reason
    #   -------------   --------- -> --------------    ----                ------
    #   True            True      -> TP  (candidate)   likely              confident + geometry-valid
    #   True            False     -> FP  (candidate)   confident_rejected  confident, geometry-invalid:<gate>
    #   False (>=LOW)   True      -> FN  (candidate)   potential           sub-threshold, geometry-valid
    #   False           False     -> TN               (not emitted)        dropped & geometry-invalid
    #   + a special FN: held-dropout (missing but flanked by confident extended) -> potential
    # ---------------------------------------------------------------------
    pipeline_keeps = detected & (pc >= PROD_THRESH)

    is_TP, is_FP, is_FN = tier_masks(pipeline_keeps, plausible, pc)

    # held-dropout: proboscis missing / very weak but flanked by confident extended
    # neighbours -> a true miss (joins the FN / potential tier)
    conf_ext = (pc >= PROD_THRESH) & extended
    prev_ext = np.zeros_like(is_FN); prev_ext[1:]  = conf_ext[:-1]
    next_ext = np.zeros_like(is_FN); next_ext[:-1] = conf_ext[1:]
    gap = (~detected) | (pc < CONF_LOW)
    is_held = gap & prev_ext & next_ext & (hc > HEAD_CONF_MIN)

    # FP triage: name the FIRST failing gate, derived straight from the gates dict
    # (so adding/renaming a gate updates the reason automatically -- no drift).
    reason_names = {
        "detected": "not_detected", "anterior": "behind_head", "in_cone": "off_cone",
        "in_reach": "too_far", "not_excursion": "excursion",
        "extended": "retracted", "inst_ok": "low_instance_score",
    }
    order = ["detected", "anterior", "in_cone", "in_reach", "not_excursion",
             "extended", "inst_ok"]
    fp_reason = np.select([~gates[k] for k in order],
                          [reason_names[k] for k in order], default="other")

    def rows(mask, classification, tier, reason=None):
        idx = np.where(mask)[0]
        if idx.size == 0:
            return None
        return pd.DataFrame({
            "frame": idx.astype(int),
            "classification": classification,
            "tier": tier,
            "reason": (fp_reason[idx] if reason is None else reason),
            "prob_conf": pc[idx],
            "dist_mm": dist[idx] / ppm,
            "off_axis_deg": np.degrees(off[idx]),
            "prob_to_abd_mm": prob_to_abd[idx] / ppm,
        })

    parts = [
        rows(is_TP,   "TP", TIER_LIKELY,    "confident_geometry_valid"),
        rows(is_FP,   "FP", TIER_FP,        None),                       # per-gate reason
        rows(is_FN,   "FN", TIER_POTENTIAL, "recoverable_below_threshold"),
        rows(is_held, "FN", TIER_POTENTIAL, "held_dropout"),
    ]
    parts = [p for p in parts if p is not None]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ==========================================================================
# resolve identity / chunk video, then write one feather per (fly, tier)
# ==========================================================================


def resolve_video_paths(df, experiment, identity):
    """Add PER-FRAME 'video_file' + 'local_frame' columns, resolved from each row's
    own frame_number. Because resolution is per-frame (not per-bout), a bout that
    STRADDLES a chunk boundary correctly gets two different video_files across its
    frames, and local_frame resets at the boundary.

    This is the ONLY place flyhostel path logic lives, so the SLEAP-env builder never
    imports flyhostel — it just reads video_file / local_frame as plain columns.
    """
    df = df.copy()
    basedir   = get_basedir(experiment)
    chunksize = get_chunksize(experiment)
    n_animals = get_number_of_animals(experiment)

    fn = df["frame_number"].astype(int)
    df["chunk"]       = fn // chunksize
    df["local_frame"] = fn %  chunksize
    df["identity"]    = identity

    if n_animals > 1:
        # local_identity varies per chunk -> resolve per (chunk, identity)
        table = get_local_identities(get_dbfile(basedir),
                                     frame_numbers=fn).reset_index(drop=True)
        key = table[["chunk", "identity", "local_identity"]].drop_duplicates()
        assert key.groupby(["chunk", "identity"]).size().max() == 1, \
            "local_identity not unique per (chunk, identity) — merge would fan out"
        df = df.merge(key, on=["chunk", "identity"], how="left")
        li = df["local_identity"].astype("Int64").astype(str).str.zfill(3)
    else:
        df["local_identity"] = 0
        li = pd.Series("000", index=df.index)

    df["video_file"] = (basedir + "/flyhostel/single_animal/"
                        + li + "/" + df["chunk"].astype(str).str.zfill(6) + ".mp4")
    return df


def explain_frame(path, params, frame_number, track=0):
    """Explain (a) why a frame lands in its tier and (b) if it reached `likely` and
    falls inside a scored bout, why that bout was labelled pe / nonPE.

    DRIFT-PROOF BY CONSTRUCTION:
      * the gates come from gate_masks()  -- the same call classify_detections makes
      * the tier comes from tier_masks()  -- the same truth table
      * the behaviour reason is READ from the `label_reason` column that label_bouts
        wrote at labelling time; it is never re-derived here.
    Nothing in this function reimplements a rule, so it cannot disagree with the
    pipeline. Adding a gate or a label rule shows up here automatically.
    """
    locs, sc, nodes, inst = load_arrays(path)
    g = compute_geometry(locs, sc, nodes, inst)
    exp = os.path.basename(path).split("__")[0]
    ppm, fps = get_pixels_per_mm(exp), get_framerate(exp)
    chunksize = get_chunksize(exp)
    first_fn = get_first_frame_number(path, chunksize)
    row = int(frame_number) - first_fn
    if row < 0 or row >= g["dist"].shape[0]:
        print(f"frame {frame_number}: row {row} out of range"); return None

    # identical computation to the pipeline, then index the row
    gates, plausible, v = gate_masks(locs, g, params, ppm, fps, track)

    dist, off, pc = v["dist"][row], v["off"][row], v["pc"][row]
    along, ac, isc = v["along"][row], v["ac"][row], v["isc"][row]
    detail = {
        "detected":      "",
        "in_reach":      f"dist={dist:.1f}px <= {v['max_ext_px']:.1f}",
        "in_cone":       (f"off={np.degrees(off):.1f}deg <= {np.degrees(v['cone_rad']):.1f}"
                          f"  axis_ok={bool(v['axis_ok'][row])} (ac={ac:.2f})"),
        "anterior":      f"along={along:+.1f}px",
        "extended":      f"{dist/ppm:.3f}mm > ext_min={v['ext_min_px']/ppm:.3f}mm",
        "not_excursion": f"excursion={bool(v['excursion'][row])}",
        "inst_ok":       f"inst={isc:.2f} >= {INST_SCORE_MIN}",
    }

    print(f"\nframe {frame_number} (row {row})  prob_conf={pc:.3f}")
    for name, mask in gates.items():
        ok = bool(mask[row])
        print(f"  {'PASS' if ok else 'FAIL'}  {name:14s} {detail[name]}")

    keeps = bool(gates["detected"][row]) and (pc >= PROD_THRESH)
    plaus = bool(plausible[row])
    is_TP, is_FP, is_FN = tier_masks(keeps, plaus, pc)
    print(f"  pipeline_keeps={keeps} (pc>={PROD_THRESH})   plausible={plaus}")

    if is_TP:
        tier = "likely"
    elif is_FP:
        tier = "confident_rejected"
        failed = [n for n, m in gates.items() if not bool(m[row])]
        print(f"  reason: confident but geometry-invalid -> {failed}")
    elif is_FN:
        tier = "potential"
        print(f"  reason: geometry-valid but pc<{PROD_THRESH} (recoverable FN)")
    else:
        tier = "TN / not-emitted"
    print(f"  --> TIER: {tier}")

    # ---- PE / nonPE stage (only `likely` frames enter the behaviour stage) ----
    if tier != "likely":
        print("  (PE/nonPE not evaluated: only `likely` frames enter the behaviour stage)")
        return tier, None

    identity = os.path.basename(path).split(".")[0].split("__")[1]
    fly=f"{exp}__{identity}"
    bouts_feather = f"pe_bouts/{fly}_pe_bouts.feather"

    b = pd.read_feather(bouts_feather)
    hit = b[(b["start_fn"] <= frame_number) & (b["end_fn"] >= frame_number)]
    if hit.empty:
        print("  this frame is in `likely` but inside no scored bout "
              "(bout segmentation didn't group it into an event)")
        return tier, None

    r = hit.iloc[0]
    is_pe = (r["label"] == "pe")
    print(f"  --> BEHAVIOUR: {'PE' if is_pe else 'nonPE'}  (label={r['label']}, "
          f"pe_score={r['pe_score']:.2f})")
    print(f"    Start {r['start_fn']} End {r['end_fn']}")
    # the reason was recorded by label_bouts -- not recomputed here
    print(f"    rule fired: {r['label_reason']}")
    print(f"    burst {int(r['burst_id'])}  n_in_burst={int(r['n_in_burst'])}  "
          f"solitary={bool(r['is_solitary'])}")
    print(f"    dur_s={r['dur_s']:.2f}  baseline={r['baseline_mm']:.3f}mm  "
          f"peak={r['peak_dist_mm']:.3f}mm  amp={r['amp_mm']:.3f}mm")
    rt = r["return_time_s"]
    print(f"    returned={bool(r['returned'])}"
          + (f" (in {rt:.2f}s)" if pd.notna(rt) else " (never)"))
    print(f"    quiescence_bout_s={r['quiescence_bout_s']:.1f}  "
          f"(before={r['quiesc_before_s']:.1f}s after={r['quiesc_after_s']:.1f}s)")
    print(f"    body_move_frac={r['body_move_frac']:.3f}  "
          f"leg_speed_med={r['leg_speed_med']:.3f}  "
          f"autocorr_peak={r['autocorr_peak']:.2f}" if pd.notna(r["autocorr_peak"])
          else f"    body_move_frac={r['body_move_frac']:.3f}  "
               f"leg_speed_med={r['leg_speed_med']:.3f}  autocorr_peak=nan")
    return tier, r["label"]


def review_file(path, params):
    out = {}
    experiment, identity = os.path.basename(path).split(".")[0].split("__")
    identity = int(identity)
    chunksize = get_chunksize(experiment)

    try:
        df = classify_detections(path, params)
    except (OSError, KeyError, ValueError) as e:
        logger.error("%s: %s", os.path.basename(path), e)
        return out
    if df.empty:
        return out

    first_fn = get_first_frame_number(path, chunksize)
    df["frame_number"] = df["frame"] + first_fn

    # per-FRAME video_file + local_frame (handles chunk-straddling bouts)
    df = resolve_video_paths(df, experiment, identity)

    # one record entry per (fly-chunk, tier). video_file/local_frame are carried
    # PER FRAME so the builder never recomputes them.
    for (chunk, tier), dft in df.groupby(["chunk", "tier"]):
        fid = f"{experiment}__{str(identity).zfill(2)}_{str(chunk).zfill(6)}"
        out[(fid, tier)] = {
            "fid": fid, "tier": tier,
            "chunksize": chunksize, "h5_path": path,
            "frames":      dft["frame_number"].to_numpy(),
            "local_frame": dft["local_frame"].to_numpy(),
            "video_file":  dft["video_file"].to_numpy(),
            "dist_mm":     dft["dist_mm"].to_numpy(),
            "prob_conf":   dft["prob_conf"].to_numpy(),
            "classification": dft["classification"].to_numpy(),
            "reason":      dft["reason"].to_numpy(),
        }
    # human-readable dump per fly (all tiers together)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.assign(fly=f"{experiment}__{str(identity).zfill(2)}").to_feather(
        f"{OUTPUT_DIR}/{experiment}__{str(identity).zfill(2)}_candidates.feather")
    return out



def proboscis_candidates_for_fly(fly):

    experiment, identity = fly.split("__")[0], int(fly.split("__")[1])
    loader=FlyHostelLoader(experiment, identity)
    pose_file=loader.get_pose_file_h5py("raw")

    check_file_contains_everything_needed(pose_file, experiment, identity)

    params = PARAMS.copy()

    _, _, nodes, _ = load_arrays(pose_file)
    pd.Series(nodes).to_csv("nodes.csv", index=False, header=["node"])
    records = {}
    records.update(
        review_file(pose_file, params)
    )

    for (fid, tier), rec in records.items():
        n = len(rec["frames"])
        assert n == len(rec["dist_mm"]) == len(rec["prob_conf"]) == len(rec["reason"]) \
                 == len(rec["video_file"]) == len(rec["local_frame"]), \
            f"{fid}/{tier}: misaligned arrays"
        df = pd.DataFrame({
            "fly_id": fid, "tier": tier,
            "chunksize": rec["chunksize"], "h5_path": rec["h5_path"],
            "frame_number": rec["frames"].astype(int),
            "local_frame": rec["local_frame"].astype(int),   # per-frame (straddle-safe)
            "video_file": rec["video_file"],                 # per-frame (straddle-safe)
            "dist_mm": rec["dist_mm"].astype(float),
            "prob_conf": rec["prob_conf"].astype(float),
            "classification": rec["classification"], "reason": rec["reason"],
        })
        os.makedirs(f"{RECORDS_DIR}/{tier}", exist_ok=True)
        df.to_feather(f"{RECORDS_DIR}/{tier}/{fid}_records.feather")
    print(f"wrote records for {len(records)} (fly,tier) groups under {RECORDS_DIR}/")



# derive parameters
def derive_parameters(files):
    dist_mm, offax, dist_for_off, vel = [], [], [], []
    for path in tqdm(files, desc="Fitting gates"):
        try:
            locs, sc, nodes, inst_scores = load_arrays(path)
            g = compute_geometry(locs, sc, nodes, inst_scores)
        except (OSError, KeyError, ValueError) as e:
            logger.warning("skip %s: %s", os.path.basename(path), e)
            continue
        exp = os.path.basename(path).split("__")[0]
        ppm, fps = get_pixels_per_mm(exp), get_framerate(exp)

        good = (g["prob_conf"] > CONF_HI_FIT) & (g["head_conf"] > CONF_HI_FIT) & (g["along"] > 0)
        good_axis = good & (g["axis_conf"] > CONF_HI_FIT)
        dmm = g["dist"] / ppm
        dist_mm.append(_subsample(dmm[good & np.isfinite(dmm)]))

        m = good_axis & np.isfinite(g["off_axis"]) & np.isfinite(dmm)
        offax.append(g["off_axis"][m]); dist_for_off.append(dmm[m])

        cp = g["prob_conf"] > CONF_HI_FIT
        pair = np.zeros_like(cp); pair[1:, :] = cp[1:, :] & cp[:-1, :]
        v = np.where(pair & np.isfinite(g["disp"]), g["disp"] / ppm * fps, np.nan)
        vel.append(_subsample(v.ravel()))

    dist_mm = np.concatenate(dist_mm)
    offax = np.concatenate(offax); dist_for_off = np.concatenate(dist_for_off)
    vel = np.concatenate(vel)

    max_ext_mm = EXT_PAD * np.percentile(dist_mm, EXT_P)
    cone_rad = CONE_PAD * np.percentile(offax[dist_for_off < max_ext_mm], CONE_P)
    max_vel = VEL_PAD * np.percentile(vel, VEL_P)
    return dict(max_ext_mm=float(max_ext_mm), cone_rad=float(cone_rad),
                max_tip_vel_mm_s=float(max_vel))



def load_and_derive_parameters(files):
    print(f"{len(files)} pose files")

    params = derive_parameters(files)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    yaml.safe_dump(params, open(PARAMS_FILE, "w"))
    print("gates:", params)
    return params



# ==========================================================================
if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--fly", required=True)
    args=ap.parse_args()
    proboscis_candidates_for_fly(args.fly)
