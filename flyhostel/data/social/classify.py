"""
PIPELINE 2, stage 2 — what kind of interaction is each bout?

Input: contact bouts (or centroid proximity bouts) after centroid.annotate_approach and
centroid.orient, i.e. with initiator / recipient columns. Every bout gets exactly one
label from a fixed, mutually exclusive cascade, plus the reason that fired:

    0. a fly untracked in the approach window             -> unclassified
    1. neither fly moving in the approach window        -> static_contact
    2. initiator moving, recipient immobile for at least
       min_pre_immobile_s, and the approach is clearly
       the initiator's (share >= approach_share_min)     -> approach_to_immobile
    3. both flies moving                                  -> mutual_encounter
    4. initiator moving, recipient still but not long
       enough / approach share ambiguous                  -> approach_to_resting

Contact geometry (pose only) is reported alongside, not folded into the label, so it can
be cross-tabulated:
    ini_part_onset / rec_part_onset   which part touched first (head / body / leg)
    contact_kind                      e.g. "leg->body" = initiator leg on recipient body
    rec_side                          where the initiator came from, in the recipient's
                                      own frame: anterior / left / right / posterior
"""
import numpy as np


LABELS = ["static_contact", "approach_to_immobile", "mutual_encounter", "approach_to_resting"]


def classify_bouts(bouts, movement_cfg, response_cfg):
    """Add ``label`` and ``label_reason`` to oriented bouts (see module docstring)."""
    df = bouts.copy()
    with np.errstate(invalid="ignore"):
        ini_moving = df["ini_speed_pre"].to_numpy() > movement_cfg.move_speed_mm_s
        rec_moving = df["rec_speed_pre"].to_numpy() > movement_cfg.move_speed_mm_s
        rec_long_still = df["rec_immobile_s"].to_numpy() >= response_cfg.min_pre_immobile_s
        clear = df["initiator_share"].to_numpy() >= response_cfg.approach_share_min

    untracked = df[["ini_speed_pre", "rec_speed_pre"]].isna().any(axis=1).to_numpy()
    conditions = [
        untracked,
        ~ini_moving & ~rec_moving,
        ini_moving & rec_long_still & clear,
        ini_moving & rec_moving,
        ini_moving,
    ]
    reasons = [
        "approach window not tracked",
        "neither fly moved in the approach window",
        f"initiator moving; recipient immobile >= {response_cfg.min_pre_immobile_s:g} s; "
        f"initiator share >= {response_cfg.approach_share_min:g}",
        "both flies moved in the approach window",
        "initiator moving; recipient still but briefly, or approach share ambiguous",
    ]
    df["label"] = np.select(conditions, ["unclassified"] + LABELS, default="unclassified")
    df["label_reason"] = np.select(conditions, reasons,
                                   default="recipient moved, initiator below the movement threshold")

    if "ini_part_onset" in df.columns:
        df["contact_kind"] = df["ini_part_onset"].astype(str) + "->" + df["rec_part_onset"].astype(str)
    return df
