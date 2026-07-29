"""
pe_features.py   —   NORMAL environment (numpy / h5py / flyhostel)

Per-EVENT features to separate sleep proboscis-extension (PE) from feeding/drinking
and grooming, and a rule-based first-pass label. Feeds a classifier (or your
wavelet+RF) once you have review labels.

DESIGN — why two feature levels, and how solitary PEs are handled:

  A PE burst is a near-periodic train of ~1 s extend-retract BOUTS spaced ~2-3 s
  apart, with the REST of the body frozen. But a burst may contain a SINGLE bout —
  one lone PE that looks identical to a bout inside a multi-bout burst. Rhythmicity
  (inter-bout periodicity) therefore CANNOT be the primary discriminator: a solitary
  PE has no inter-bout interval.

  So the discriminating weight is on features EVERY bout has:
      * bout SHAPE      — a clean ~1 s extend-retract that returns to baseline
                          (vs a feeding PLATEAU that stays out for seconds-minutes,
                           vs irregular grooming)
      * body/leg CONTRAST — proboscis moving while thorax & forelegs are static
                          (grooming moves the forelegs; feeding may add walking)
  Rhythmicity (autocorr side-peak, IBI regularity) is computed too, but only BOOSTS
  confidence when neighbours exist; it is NaN/absent for solitary bouts, which are
  still labelled from shape + movement. Every bout — solitary or not — gets a row.
"""

import ast
import os
import yaml
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from flyhostel.data.pose.main import FlyHostelLoader

from flyhostel.data.pose.landmarks import distance_from_points_to_ellipse
from flyhostel.utils.pose_export import load_arrays, get_first_frame_number, load_frame_numbers
from .proboscis_candidates import (
    compute_geometry, EXT_FRAC, PROD_THRESH, PARAMS,
    resolve_video_paths, gate_masks
)


# ==========================================================================
# ||||||||||||||||||  MANUAL PARAMETERS (not data-derived)  ||||||||||||||||||
# ==========================================================================
TRACK          = 0

# --- extension signal cleaning / bout segmentation ---
BRIDGE_S       = 0.10   # bridge dropouts shorter than this inside a bout.  unit: s
                        #   (repairs single-frame misses; never merges the ~2-3 s inter-bout gap)

# --- bout-shape bands defining a PE-like extend-retract ---
PE_DUR_MIN_S   = 0.30   # a PE bout lasts roughly ~1 s; accept this..            unit: s
PE_DUR_MAX_S   = 2.00   # ..to this. Longer contiguous extension = plateau.      unit: s
PLATEAU_DUR_S  = 3.00   # a single bout longer than this is a held extension =
                        #   feeding/drinking, not PE.                            unit: s

# --- RETURN TO BASELINE (the PE shape signature; RELATIVE, no amplitude cutoff) ---
# A real PE goes out and comes BACK to where the proboscis was before it started,
# within a few seconds. This is scale-free: it compares each bout to its OWN
# baseline, so a small extension and a large one are judged the same way. There is
# deliberately NO absolute "how far must it extend" threshold.
RETURN_WINDOW_S = 3.00  # retraction must COMPLETE within this many seconds of the   unit: s
                        #   moment the extension starts (bout start).
RETURN_FRAC     = 0.25  # "returned" = dist falls back to within RETURN_FRAC of the  unit: fraction
                        #   bout's own amplitude above its baseline, i.e.
                        #   dist <= baseline + RETURN_FRAC * (peak - baseline).
                        #   0.25 = came back down through 75% of what it went up.

# --- QUIESCENCE BOUT containment (drops feeding wedged between walking bouts) ---
MIN_QUIESCENCE_S = 10.0 # the PE must sit inside a CONTIGUOUS quiescence bout at    unit: s
                        #   least this long. Feeding often happens in a brief still
                        #   window right after the fly stops walking and just before
                        #   it resumes; a 10 s stillness requirement removes those.
MOVE_BRIDGE_S    = 0.10 # ignore movement blips shorter than this when building      unit: s
                        #   quiescence bouts (tracking jitter would otherwise
                        #   fragment a long, genuinely still stretch).

# --- movement contrast (the strongest single discriminator) ---
BODY_MOVE_MM_S = 1.00   # thorax speed above this = body moving.                 unit: mm/s
                        #   (CALIBRATE per rig — 1 px jitter is a different mm/s on each.)
LEG_MOVE_MM_S  = 2.00   # foreleg speed (relative to thorax) above this = legs   unit: mm/s
                        #   active => grooming. (CALIBRATE against known grooming.)


REAR_LEG_MOVE_MM_S = 3.0   # rear-leg speed above this = legs active -> grooming/moving.
                           # CALIBRATE separately: rear legs may have a different baseline
                           # jitter than forelegs. Histogram rear_leg_speed_med for
                           # confirmed groom vs pe.

BODY_MOVE_FRAC = 0.20   # if > this fraction of a bout's frames have the body
                        #   moving, the event is contaminated by locomotion.     unit: fraction

# --- burst grouping / rhythmicity (context; graceful when solitary) ---
BURST_MAX_GAP_S = 20.00  # bouts closer than this belong to one burst.            unit: s
                        #   (inter-bout interval ~1 s, so 2 s groups a burst but
                        #    separates distinct bursts.)
CONTEXT_WIN_S   = 8.00  # window (centred on the bout) for autocorr / neighbour  unit: s
                        #   counting. Must exceed one inter-bout interval.
IBI_LO_S, IBI_HI_S = 1.5, 4.0  # autocorr side-peak search band (inter-bout).    unit: s

BASELINE_WIN_S  = 1.00  # window before a bout used to estimate the retracted    unit: s
                        #   baseline for amplitude.

FORELEGS = ("fRL", "fLL")   # nodes used for the grooming (leg-motion) contrast
REARLEGS = ("rRL", "rLL")  
# --- FOOD RING proximity (feeding / drinking discriminator) ---
# Flies feed/drink by placing the proboscis tip essentially ON the perimeter of the
# food blob. Distance is signed (negative = inside the ellipse).
FOOD_RING_MM   = 2   # proboscis within this of the food boundary -> feeding.   unit: mm
                        #   (CALIBRATE: histogram prob_food_min_mm for hand-labelled
                        #    feed vs pe; put the cut in the gap.)
NEAR_FOOD_MODE = "ring" # "ring"           : |signed distance| <= FOOD_RING_MM
                        #                    (near the perimeter, in or out)
                        # "inside_or_ring" : signed distance <= FOOD_RING_MM
                        #                    (also counts the proboscis DEEP INSIDE the
                        #                     blob as feeding -- use this if flies stand
                        #                     on the food and probe well inside it)

# --- FORELEG-to-PROBOSCIS proximity (grooming discriminator) ---
# Proboscis grooming draws the proboscis through the forelegs, so the foreleg tips come
# very close to the proboscis tip. During PE the legs are tucked and the proboscis
# extends away from them.
LEG_NEAR_PROB_MM = -1 # foreleg tip closer than this to the proboscis tip -> groom.  unit: mm
                        #   (CALIBRATE: histogram leg_prob_min_mm for known groom vs pe.
                        #    NB in a TOP view a folded foreleg can sit near the head, so
                        #    this threshold is genuinely data-dependent.)

USE_FOOD_RULE = True    # set False to disable the food-ring rule entirely
USE_LEG_PROXIMITY_RULE = False   # foreleg TIP near proboscis tip -> groom
USE_LEG_MOTION_RULE     = False   # fore/rear leg SPEED high        -> groom


PE_CONF_MIN = 0.65    # peak proboscis confidence must clear this (well above PROD_THRESH=0.5)
                      # CALIBRATE: histogram conf_at_peak for GUI-confirmed pe vs false-pe
USE_CONF_RULE = True    # gate PE on peak proboscis confidence (conf_at_peak >= PE_CONF_MIN)

# ==========================================================================
# signals: gated extension (mm) + body & leg speed (mm/s), full length
# ==========================================================================

def project_thorax_to_arena(loader, thorax, frame_numbers):
    square_width=loader.square_width
    square_height=loader.square_height
    thorax_arena_xy=thorax.copy()
    thorax_arena_xy[:, 0] -= square_width//2
    thorax_arena_xy[:, 1] -= square_height//2
    
    centroid_coords=pd.DataFrame({"frame_number": frame_numbers}).merge(loader.dt[["frame_number", "center_x", "center_y"]], on="frame_number", how="left")
    assert thorax_arena_xy.shape[0]==centroid_coords.shape[0]
    
    # TODO what happens if for some frame center_x or center_y are nan?

    thorax_arena_xy[:, 0]+= centroid_coords["center_x"]
    thorax_arena_xy[:, 1]+= centroid_coords["center_y"]
    return thorax_arena_xy



def signals_from_h5(loader, params, track=TRACK):
    """Per-frame signals, built from the SAME gate classify_detections uses."""
    loader.load_centroid_data(cache="/flyhostel_data/cache")
    assert loader.dt is not None
    loader.load_landmarks()
    path = loader.get_pose_file_h5py("raw")

    locs, sc, nodes, inst = load_arrays(path)
    frame_numbers = load_frame_numbers(path, loader.chunksize)
    g = compute_geometry(locs, sc, nodes, inst)
    exp = os.path.basename(path).split("__")[0]
    ppm, fps = loader.pixels_per_mm, loader.framerate
    ext_min_mm = EXT_FRAC * params["max_ext_mm"]

    # ---------------------------------------------------------------------
    # EXTENSION TRACE: the shared geometry gate + the CONFIDENCE cut.
    # Without `pc >= PROD_THRESH`, sub-threshold hallucinations (proboscis not
    # visible; the node parked near the head) enter dist_mm as real numbers and
    # fuse consecutive PE waves into one multi-second "bout".
    # 'extended' is deliberately EXCLUDED: it is a segmentation threshold, not a
    # validity test, and we need the sub-extension frames to see the baseline.
    # ---------------------------------------------------------------------
    gates, _, v = gate_masks(locs, g, params, ppm, fps, track)
    sig_mask = (v["pc"] >= PROD_THRESH)
    for name, m in gates.items():
        if name == "extended":
            continue
        sig_mask = sig_mask & m
    dist_mm = np.where(sig_mask, v["dist"] / ppm, np.nan)

    ti = nodes.index("thorax")
    thorax = locs[:, ti, :, track]                     # CROP frame
    thorax_arena_xy = project_thorax_to_arena(loader, thorax, frame_numbers)

    # ---------------------------------------------------------------------
    # BODY SPEED must come from ARENA coords. The crop is CENTRED on the fly, so
    # in crop coords the thorax barely moves even while the fly walks -- crop-frame
    # speed is ~0 always, which is why quiescence_bout_s came out at 3035 s.
    # ---------------------------------------------------------------------
    body_speed = np.concatenate(
        [[np.nan], np.linalg.norm(np.diff(thorax_arena_xy, axis=0), axis=1)]) / ppm * fps

    # leg speed stays CROP-relative (leg - thorax): pure limb motion, no arena needed
    leg_sp = []
    for name in FORELEGS:
        if name in nodes:
            rel = locs[:, nodes.index(name), :, track] - thorax
            leg_sp.append(np.concatenate(
                [[np.nan], np.linalg.norm(np.diff(rel, axis=0), axis=1)]) / ppm * fps)
    leg_speed = (np.nanmax(np.stack(leg_sp), axis=0) if leg_sp
                 else np.full_like(body_speed, np.nan))
    rear_sp = []
    for name in REARLEGS:
        if name in nodes:
            rel = locs[:, nodes.index(name), :, track] - thorax
            rear_sp.append(np.concatenate(
                [[np.nan], np.linalg.norm(np.diff(rel, axis=0), axis=1)]) / ppm * fps)
    rear_leg_speed = (np.nanmax(np.stack(rear_sp), axis=0) if rear_sp
                    else np.full_like(body_speed, np.nan))
        
    leg_prob_mm = foreleg_proboscis_distance(locs, nodes, track, ppm)

    # --- proboscis -> food boundary -------------------------------------------
    food_blobs = loader.landmarks.loc[loader.landmarks["shape"] == "food"]
    if len(food_blobs) == 0 or thorax_arena_xy is None:
        prob_food_mm = np.full(dist_mm.size, np.nan)   # feature off -> never rejects
    else:
        pi = nodes.index("proboscis")
        prob_arena = pose_to_arena_xy(locs[:, pi, :, track], thorax, thorax_arena_xy)

        per_blob = []
        for _, blob in food_blobs.iterrows():
            spec = ast.literal_eval(blob["specification_norm"])   # not eval()
            # normalize to what proboscis_food_distance expects: cx, cy, a, b, theta(rad)
            ellipse = dict(
                cx=spec["center"][0] * loader.roi_size,
                cy=spec["center"][1] * loader.roi_size,
                a=spec["axes"][0] * loader.roi_size,
                b=spec["axes"][1] * loader.roi_size,
                theta=np.radians(spec["angle"]),
            )
            per_blob.append(proboscis_food_distance(prob_arena, ellipse, ppm))

        per_blob = np.vstack(per_blob)                 # (n_blobs, n_frames), signed
        # nearest blob, NaN-safe and MODE-AWARE: 'ring' ranks by |d| (a blob 0.2mm
        # outside beats one 10mm inside); 'inside_or_ring' ranks by signed d.
        key = np.abs(per_blob) if NEAR_FOOD_MODE == "ring" else per_blob
        key = np.where(np.isfinite(key), key, np.inf)  # NaN can never win
        nearest = np.argmin(key, axis=0)
        prob_food_mm = np.take_along_axis(per_blob, nearest[None, :], axis=0)[0]
        assert prob_food_mm.size == thorax_arena_xy.shape[0]

    return dict(dist_mm=dist_mm, prob_conf=v["pc"], body_speed=body_speed, leg_speed=leg_speed, rear_leg_speed=rear_leg_speed,
                leg_prob_mm=leg_prob_mm, prob_food_mm=prob_food_mm,
                ext_min_mm=ext_min_mm, fps=fps, ppm=ppm, n_frames=dist_mm.size,
                first_fn=get_first_frame_number(path, loader.chunksize))

# ==========================================================================
# bout + burst segmentation
# ==========================================================================
def _bridge(mask, bridge):
    m = np.asarray(mask, bool).copy()
    if bridge <= 0:
        return m
    diff = np.diff(np.concatenate([[True], m, [True]]).astype(np.int8))
    for s, e in zip(np.where(diff == -1)[0], np.where(diff == 1)[0]):
        if s > 0 and e < len(m) and (e - s) <= bridge:
            m[s:e] = True
    return m


def segment_bouts(sig):
    dist_mm, fps = sig["dist_mm"], sig["fps"]
    extended = _bridge(np.isfinite(dist_mm) & (dist_mm > sig["ext_min_mm"]),
                       max(1, round(BRIDGE_S * fps)))
    d = np.diff(np.concatenate([[0], extended.view(np.int8), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]     # ends exclusive
    return list(zip(starts.tolist(), ends.tolist()))


def group_bursts(bouts, fps):
    """Assign a burst id to each bout. A burst may contain ONE bout (solitary PE)."""
    if not bouts:
        return []
    gap = round(BURST_MAX_GAP_S * fps)
    burst_id, bid = [0], 0
    for i in range(1, len(bouts)):
        if bouts[i][0] - bouts[i - 1][1] > gap:
            bid += 1
        burst_id.append(bid)
    return burst_id


def pose_to_arena_xy(node_xy, thorax_pose_xy, thorax_arena_xy):
    """Map per-frame pose coords (CROP frame) into ARENA coords.

    CRITICAL: the pose .h5 is computed on the per-animal cropped videos, so its
    coordinates are local to that crop. The food ellipse lives in ARENA coordinates.
    We recover the per-frame crop offset by anchoring on a node whose arena position
    we independently know (the thorax / tracked centroid):

        offset          = thorax_arena - thorax_pose
        arena_node_xy   = pose_node_xy + offset

    All three inputs must be in the SAME units (px or mm) -- mixing px pose with mm
    centroids silently produces nonsense.
    """
    offset = np.asarray(thorax_arena_xy) - np.asarray(thorax_pose_xy)   # (N, 2)
    return np.asarray(node_xy) + offset


def proboscis_food_distance(prob_arena_xy, food_ellipse, ppm):
    """Signed distance (mm) from the proboscis tip to the food-blob boundary.
    Negative = proboscis is INSIDE the blob. NaN where the proboscis is undetected."""
    if food_ellipse is None:
        return np.full(len(prob_arena_xy), np.nan)
    pts = np.asarray(prob_arena_xy, dtype=float)
    ok = np.isfinite(pts).all(axis=1)
    d = np.full(pts.shape[0], np.nan)
    if ok.any():
        d[ok] = distance_from_points_to_ellipse(
            pts[ok],
            food_ellipse["cx"], food_ellipse["cy"],
            food_ellipse["a"],
            food_ellipse["b"],
            food_ellipse["theta"]
        )
    return d / ppm                                   # px -> mm


def near_food_mask(prob_food_mm, mode=NEAR_FOOD_MODE, cutoff=FOOD_RING_MM):
    """Is the proboscis at the food? See NEAR_FOOD_MODE for the two readings."""
    if mode == "ring":
        return np.abs(prob_food_mm) <= cutoff        # on the perimeter (in or out)
    elif mode == "inside_or_ring":
        return prob_food_mm <= cutoff                # perimeter OR deep inside the blob
    raise ValueError(f"unknown NEAR_FOOD_MODE={mode!r}")


def foreleg_proboscis_distance(locs, nodes, track, ppm):
    """Min distance (mm) from either foreleg tip to the proboscis tip, per frame.
    Computed in the CROP frame -- it's a relative distance, so no arena transform
    is needed (and none should be applied)."""
    pi = nodes.index("proboscis")
    prob = locs[:, pi, :, track]
    dists = []
    for name in FORELEGS:
        if name in nodes:
            leg = locs[:, nodes.index(name), :, track]
            dists.append(np.linalg.norm(leg - prob, axis=1) / ppm)
    if not dists:
        return np.full(locs.shape[0], np.nan)
    return np.nanmin(np.stack(dists), axis=0)


def quiescence_runs(body_speed, fps):
    """Segment the recording into CONTIGUOUS quiescence bouts (body not moving).

    Returns per-frame arrays:
      quiescent   : bool, frame is inside a quiescence bout
      run_len_s   : float, duration (s) of the quiescence bout containing the frame
                    (0 where the frame is not quiescent)
      run_start   : int, first frame index of that bout (-1 if not quiescent)
      run_end     : int, last  frame index of that bout (-1 if not quiescent)

    NaN body speed (thorax dropout) is treated as MOVING -- conservative, so an
    untracked stretch never manufactures a fake long quiescence bout. Short movement
    blips are bridged so jitter doesn't fragment a genuinely still stretch.
    """
    moving = np.nan_to_num(body_speed, nan=np.inf) > BODY_MOVE_MM_S
    quiescent = ~moving
    # bridge brief moving blips (= short False runs inside the quiescent mask)
    quiescent = _bridge(quiescent, max(1, round(MOVE_BRIDGE_S * fps)))

    n = quiescent.size
    run_len_s = np.zeros(n, float)
    run_start = np.full(n, -1, int)
    run_end   = np.full(n, -1, int)

    d = np.diff(np.concatenate([[0], quiescent.view(np.int8), [0]]))
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):   # e exclusive
        run_len_s[s:e] = (e - s) / fps
        run_start[s:e] = s
        run_end[s:e]   = e - 1
    return quiescent, run_len_s, run_start, run_end


def returns_to_baseline(dist_mm, s, pk, baseline, peak_mm, fps):
    """Did the proboscis come BACK to where it was, within RETURN_WINDOW_S of the
    moment the extension started?  Relative test -- no absolute amplitude cutoff.

    Returns (returned: bool, return_time_s: float or nan).

    NOTE on NaN: a fully retracted proboscis is often not detected at all, so a NaN
    right after the peak is EVIDENCE of retraction, not absence of it. We therefore
    count a NaN frame as 'returned'. (Risk: a detection dropout mid-extension would
    also read as a return. The instance-score/geometry gates upstream make that rare,
    and the held-dropout logic already flags the pattern.)
    """
    if not np.isfinite(baseline) or not np.isfinite(peak_mm):
        return False, np.nan
    amp = peak_mm - baseline
    if amp <= 0:
        return False, np.nan

    thresh = baseline + RETURN_FRAC * amp          # came back down through most of it
    win_end = min(dist_mm.size, s + int(round(RETURN_WINDOW_S * fps)) + 1)
    if pk >= win_end:
        return False, np.nan

    seg = dist_mm[pk:win_end]
    back = (~np.isfinite(seg)) | (seg <= thresh)   # NaN == no longer visibly extended
    idx = np.flatnonzero(back)
    if idx.size == 0:
        return False, np.nan                       # never came back inside the window
    t_ret = pk + int(idx[0])
    return True, (t_ret - s) / fps


def _autocorr_sidepeak(x, fps):
    x = np.nan_to_num(x - np.nanmean(x)) if np.isfinite(x).any() else np.zeros_like(x)
    if x.size < 4 or not np.any(x):
        return np.nan
    ac = np.correlate(x, x, mode="full")[x.size - 1:]
    ac = ac / (ac[0] + 1e-9)
    lo, hi = int(IBI_LO_S * fps), int(IBI_HI_S * fps)
    band = ac[lo:min(hi, ac.size)]
    return float(band.max()) if band.size else np.nan


# ==========================================================================
# per-bout features (every bout gets a row, solitary included)
# ==========================================================================
def load_tier_frames(fly_id, tier="likely"):
    """Set of global frame_numbers classified into `tier` for this fly."""
    import glob
    frames = set()
    for f in glob.glob(f"records/{tier}/{fly_id}_*records.feather"):
        frames |= set(pd.read_feather(f)["frame_number"].astype(int).tolist())
    return frames


def bout_features(loader, params, track=TRACK, tier_frames=None):
    sig = signals_from_h5(loader, params, track)
    dist_mm, body, leg = sig["dist_mm"], sig["body_speed"], sig["leg_speed"]
    leg_prob, prob_food = sig["leg_prob_mm"], sig["prob_food_mm"]
    fps, first_fn, N = sig["fps"], sig["first_fn"], sig["n_frames"]

    bouts = segment_bouts(sig)
    if not bouts:
        return pd.DataFrame()
    burst_id = group_bursts(bouts, fps)
    burst_size = pd.Series(burst_id).value_counts().to_dict()

    # quiescence bouts, computed ONCE on the full body-speed trace
    quiescent, q_len_s, q_start, q_end = quiescence_runs(body, fps)

    base_w = max(1, round(BASELINE_WIN_S * fps))
    ctx_w  = max(1, round(CONTEXT_WIN_S * fps))
    centers = np.array([(s + e) // 2 for s, e in bouts])

    rows = []
    for i, (s, e) in enumerate(tqdm(bouts)):
        seg = dist_mm[s:e]

        pk = s + int(np.nanargmax(seg)) if np.isfinite(seg).any() else s


        # tier gate: keep this bout only if it is anchored in the tier of interest.
        # 'anchored' = its peak frame is a tier frame (robust; the peak is the
        # representative you'd review). Use overlap-fraction if you prefer.
        in_tier = (tier_frames is None) or (int(pk + first_fn) in tier_frames)
        if not in_tier:
            continue
    
        baseline = np.nanmedian(dist_mm[max(0, s - base_w):s])
        baseline = 0.0 if not np.isfinite(baseline) else baseline
        peak_mm = float(np.nanmax(seg)) if np.isfinite(seg).any() else np.nan

        # --- shape (available for EVERY bout, incl. solitary) ---
        dur_s = (e - s) / fps
        rise_s, fall_s = (pk - s) / fps, (e - pk) / fps
        symmetry = min(rise_s, fall_s) / max(rise_s, fall_s, 1e-6)

        # --- RETURN TO BASELINE: relative, no amplitude cutoff (item 2) ---
        returned, return_time_s = returns_to_baseline(
            dist_mm, s, pk, baseline, peak_mm, fps)

        # --- QUIESCENCE containment: is the whole bout inside ONE long still bout? (item 3)
        # require the bout's start AND end to fall in the same quiescence run, so a
        # bout straddling the start/end of stillness (fly just stopped / about to walk)
        # is not credited with the run's full duration.
        e_last = max(s, e - 1)
        same_run = (quiescent[s] and quiescent[e_last] and q_start[s] == q_start[e_last])
        if same_run:
            quiescence_bout_s = float(q_len_s[s])
            quiesc_before_s = (s - q_start[s]) / fps        # stillness already accrued
            quiesc_after_s  = (q_end[s] - e_last) / fps     # stillness still to come
        else:
            quiescence_bout_s = 0.0
            quiesc_before_s = quiesc_after_s = 0.0

        # --- FOOD RING: closest the proboscis got to the food boundary in this bout ---
        # min over the bout (any frame touching the ring is evidence of feeding), plus
        # the value at the peak (when the proboscis is maximally extended).
        pf_seg = prob_food[s:e]
        if NEAR_FOOD_MODE == "ring":
            prob_food_min_mm = float(np.nanmin(np.abs(pf_seg))) if np.isfinite(pf_seg).any() else np.nan
        else:
            prob_food_min_mm = float(np.nanmin(pf_seg)) if np.isfinite(pf_seg).any() else np.nan
        prob_food_at_peak_mm = float(prob_food[pk]) if np.isfinite(prob_food[pk]) else np.nan

        # --- GROOMING: closest a foreleg tip got to the proboscis tip in this bout ---
        lp_seg = leg_prob[s:e]
        leg_prob_min_mm = float(np.nanmin(lp_seg)) if np.isfinite(lp_seg).any() else np.nan

        # --- movement contrast during the bout ---
        bseg, lseg = body[s:e], leg[s:e]
        body_move_frac = float(np.nanmean(bseg > BODY_MOVE_MM_S)) if bseg.size else np.nan
        leg_med = float(np.nanmedian(lseg)) if np.isfinite(lseg).any() else np.nan
        body_med = float(np.nanmedian(bseg)) if np.isfinite(bseg).any() else np.nan

        # --- rear-leg motion (grooming even when forelegs are still) ---
        rlseg = sig["rear_leg_speed"][s:e]
        rear_leg_med = float(np.nanmedian(rlseg)) if np.isfinite(rlseg).any() else np.nan

        # --- proboscis confidence in this bout ---
        pc_seg = sig["prob_conf"][s:e]
        conf_at_peak = float(sig["prob_conf"][pk]) if np.isfinite(sig["prob_conf"][pk]) else np.nan
        conf_med_in_bout = float(np.nanmedian(pc_seg)) if np.isfinite(pc_seg).any() else np.nan
        conf_frac = float(np.nanmean(pc_seg >= PROD_THRESH)) if pc_seg.size else np.nan
        
        # --- burst context (graceful for solitary: n=1, IBI=NaN, autocorr low) ---
        b = burst_id[i]
        n_in_burst = burst_size[b]
        near = np.abs(centers - centers[i]) <= (ctx_w // 2)
        n_ctx = int(near.sum())
        ibi_prev = (bouts[i][0] - bouts[i - 1][1]) / fps if i > 0 and burst_id[i - 1] == b else np.nan
        ibi_next = (bouts[i + 1][0] - bouts[i][1]) / fps if i < len(bouts) - 1 and burst_id[i + 1] == b else np.nan
        w0, w1 = max(0, centers[i] - ctx_w // 2), min(N, centers[i] + ctx_w // 2)
        win = np.where(np.isfinite(dist_mm[w0:w1]), dist_mm[w0:w1], 0.0)
        ac_peak = _autocorr_sidepeak(win, fps)
        ext_frac_win = float(np.nanmean(np.isfinite(dist_mm[w0:w1]) &
                                        (dist_mm[w0:w1] > sig["ext_min_mm"])))

        rows.append(dict(
            frame_number=int(pk + first_fn), start_fn=int(s + first_fn), end_fn=int(e - 1 + first_fn),
            burst_id=int(b), n_in_burst=int(n_in_burst), is_solitary=(n_in_burst == 1),
            dur_s=dur_s, peak_dist_mm=peak_mm, baseline_mm=baseline,
            amp_mm=peak_mm - baseline,
            rise_s=rise_s, fall_s=fall_s, symmetry=symmetry,
            returned=bool(returned), return_time_s=return_time_s,
            quiescence_bout_s=quiescence_bout_s,
            quiesc_before_s=quiesc_before_s,
            quiesc_after_s=quiesc_after_s,
            prob_food_min_mm=prob_food_min_mm,
            prob_food_at_peak_mm=prob_food_at_peak_mm,
            leg_prob_min_mm=leg_prob_min_mm,
            leg_speed_med=leg_med,
            rear_leg_speed_med=rear_leg_med,
            body_move_frac=body_move_frac,
            body_speed_med=body_med,
            n_ctx=n_ctx, ibi_prev_s=ibi_prev, ibi_next_s=ibi_next,
            autocorr_peak=ac_peak, ext_frac_window=ext_frac_win,
            conf_at_peak=conf_at_peak,
            conf_med_in_bout=conf_med_in_bout,
            conf_frac=conf_frac,
            in_tier=True
        ))
        
    return pd.DataFrame(rows)


# ==========================================================================
# rule-based first-pass label (interpretable; replace with RF once labelled)
# ==========================================================================
def label_bouts(df):
    """Label each bout pe / feed / held / groom / walk / brief_quiescence / ambiguous.

    A bout is 'pe' only if ALL of:
      * it RETURNS TO BASELINE within RETURN_WINDOW_S of the extension starting
        (relative test -- no cutoff on HOW FAR the proboscis extends), and
      * it sits inside a contiguous quiescence bout >= MIN_QUIESCENCE_S (drops feeding
        wedged into the brief stillness between walking bouts), and
      * legs and body are static during it (grooming / locomotion excluded), and
      * its duration is PE-scale.

    Solitary bouts are labelled from SHAPE + MOVEMENT alone; rhythmicity only adds a
    confidence score, never gates.
    """
    lab = np.full(len(df), "ambiguous", dtype=object)

    static_body = (df["body_move_frac"] < BODY_MOVE_FRAC)
    
    static_legs = (df["leg_speed_med"] < LEG_MOVE_MM_S) & \
                    (df["rear_leg_speed_med"].fillna(0) < REAR_LEG_MOVE_MM_S)
    legs_moving = ~static_legs

    pe_shape  = df["dur_s"].between(PE_DUR_MIN_S, PE_DUR_MAX_S)
    plateau   = (df["dur_s"] > PLATEAU_DUR_S)
    returned  = df["returned"].astype(bool)
    long_quiet = (df["quiescence_bout_s"] >= MIN_QUIESCENCE_S)
    confident = (df["conf_at_peak"].fillna(0) >= PE_CONF_MIN)
    
    # order matters: most specific first. Each rule ALSO records why it fired, so
    # explain_frame can report the reason without re-deriving (and drifting from) it.
    reason = np.full(len(df), "no rule matched", dtype=object)

    # NaN-safe: a missing feature must NEVER reject a bout (fillna -> "far away").
    # NB pass mode/cutoff EXPLICITLY: near_food_mask's defaults are bound at import
    # time, so relying on them would ignore any later change to the module constants.
    near_food = near_food_mask(df["prob_food_min_mm"].to_numpy(dtype=float),
                               mode=NEAR_FOOD_MODE, cutoff=FOOD_RING_MM) if USE_FOOD_RULE else np.zeros(len(df), bool)
    near_food = pd.Series(np.where(df["prob_food_min_mm"].isna(), False, near_food),
                          index=df.index)

    legs_near = (df["leg_prob_min_mm"].fillna(np.inf) <= LEG_NEAR_PROB_MM) \
                if USE_LEG_PROXIMITY_RULE else pd.Series(False, index=df.index)

    legs_moving = (~static_legs) if USE_LEG_MOTION_RULE else pd.Series(False, index=df.index)

    m = (~plateau & ~near_food & (legs_near | legs_moving)).to_numpy()
    lab[m] = "groom"
    reason[m] = (f"foreleg within LEG_NEAR_PROB_MM ({LEG_NEAR_PROB_MM}mm) of proboscis "
                 f"or leg_speed_med >= LEG_MOVE_MM_S ({LEG_MOVE_MM_S})")


    lab[plateau.to_numpy()] = "feed"
    reason[plateau.to_numpy()] = f"dur_s > PLATEAU_DUR_S ({PLATEAU_DUR_S}s) -> held extension"

    # proboscis parked on the food boundary -> feeding / drinking, whatever the shape
    m = (~plateau & near_food).to_numpy()
    lab[m] = "feed"
    reason[m] = (f"proboscis within FOOD_RING_MM ({FOOD_RING_MM}mm) of the food blob "
                 f"[{NEAR_FOOD_MODE}] -> feeding/drinking")


    clean = ~plateau & ~near_food & ~legs_near & static_legs
    m = (clean & ~static_body).to_numpy()
    lab[m] = "walk";   reason[m] = f"body_move_frac >= BODY_MOVE_FRAC ({BODY_MOVE_FRAC}) -> locomotion"

    still = clean & static_body
    m = (still & ~returned).to_numpy()
    lab[m] = "held";   reason[m] = f"never returned to baseline within RETURN_WINDOW_S ({RETURN_WINDOW_S}s)"

    m = (still & returned & ~long_quiet).to_numpy()
    lab[m] = "brief_quiescence"
    reason[m] = f"quiescence_bout_s < MIN_QUIESCENCE_S ({MIN_QUIESCENCE_S}s) -> feeding between walks"

    # m = (still & returned & long_quiet & pe_shape).to_numpy()
    # lab[m] = "pe"
    # reason[m] = ("returned to baseline + long quiescence + still body/legs + "
    #              "away from food + legs off proboscis + PE-scale duration")

    m = (still & returned & long_quiet & ~pe_shape).to_numpy()
    reason[m] = f"dur_s outside PE band [{PE_DUR_MIN_S}, {PE_DUR_MAX_S}]s"


    if USE_CONF_RULE:
        confident = (df["conf_at_peak"].fillna(0) >= PE_CONF_MIN)
    else:
        confident = pd.Series(True, index=df.index)   # confidence never rejects

    pe_ok = still & returned & long_quiet & pe_shape
    m = (pe_ok & confident).to_numpy()
    lab[m] = "pe"
    reason[m] = ("returned to baseline + long quiescence + still body/legs + "
                "away from food + PE-scale duration"
                + (f" + conf_at_peak >= PE_CONF_MIN ({PE_CONF_MIN})" if USE_CONF_RULE else ""))

    if USE_CONF_RULE:
        m = (pe_ok & ~confident).to_numpy()
        lab[m] = "low_conf"
        reason[m] = f"conf_at_peak < PE_CONF_MIN ({PE_CONF_MIN}) -> proboscis barely detected"


    df = df.copy()
    df["label"] = lab
    df["label_reason"] = reason
    df["near_food"] = near_food
    df["legs_near_proboscis"] = legs_near

    # continuous PE confidence: shape + return + stillness + away-from-food/legs
    stillness = (1 - df["body_move_frac"].clip(0, 1)) * static_legs.astype(float)
    quiet_q = (df["quiescence_bout_s"] / MIN_QUIESCENCE_S).clip(0, 1)
    rhythm = df["autocorr_peak"].fillna(0).clip(0, 1)                      # 0 for solitary
    away = (~near_food).astype(float) * (~legs_near).astype(float)         # not feeding/grooming
    df["pe_score"] = (0.25 * pe_shape.astype(float)
                      + 0.20 * returned.astype(float)
                      + 0.20 * away
                      + 0.15 * quiet_q
                      + 0.10 * stillness
                      + 0.10 * rhythm).clip(0, 1)
    return df


def pe_features_for_fly(fly):

    params = PARAMS.copy()

    experiment, identity=fly.split("__")
    identity=int(identity)
    tier_frames = load_tier_frames(fly, "likely")
    loader=FlyHostelLoader(experiment, identity)
    pose_file=loader.get_pose_file_h5py("raw")

    try:
        df = label_bouts(bout_features(loader, params, tier_frames=tier_frames))
    except (OSError, KeyError, ValueError) as e:
        print(f"  skip {os.path.basename(pose_file)}: {e}")
        return
    if df.empty:
        return
    experiment, identity = fly.split("__")
    # scalars the builder needs, carried as columns so it imports no flyhostel
    df["fly"] = fly
    df["h5_path"] = pose_file
    df["chunksize"] = loader.chunksize
    df["fps"] = loader.framerate
    # per-frame video_file + local_frame for the bout's PEAK frame (frame_number).
    # resolve_video_paths keys on frame_number, so the peak lands in the right
    # chunk video even when its bout straddles a boundary.
    df = resolve_video_paths(df, experiment, int(identity))

    df = df.sort_values(["burst_id", "start_fn"]).reset_index(drop=True)

    df["bout_in_burst"] = (
        df.groupby("burst_id")
        .cumcount()
    )

    df["bout_uid"] = (
        df["burst_id"] * 1000
        + df["bout_in_burst"]
    )


    os.makedirs("pe_bouts", exist_ok=True)
    df.to_feather(f"pe_bouts/{fly}_pe_bouts.feather")
    vc = df["label"].value_counts().to_dict()
    n_solo_pe = int(((df.label == "pe") & df.is_solitary).sum())
    print(f"  {df['fly'].iloc[0]}: {vc}  (solitary PE: {n_solo_pe})")
    return df


# ==========================================================================
if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--fly", required=True)
    ap.add_argument("--n-jobs", type=int, default=1)
    args=ap.parse_args()

    out=pe_features_for_fly(args.fly)
    
    print(f"\n{len(out)} bouts over {out['fly'].nunique()} flies -> pe_bouts.feather")
    print(out["label"].value_counts())