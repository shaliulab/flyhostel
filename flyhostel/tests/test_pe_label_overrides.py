"""
Tests for the targets of the override rule of the proboscis extension pipeline (label_overrides.py):
in arenas the whole arena is food, so the rule may also rescue bouts labelled feed
"""
import pandas as pd

from flyhostel.data.pose.proboscis_extension.label_overrides import (
    ARENA_OVERRIDE_TARGET_LABELS,
    OVERRIDE_TARGET_LABELS,
    label_composition,
    override_target_labels,
)


def test_arenas_also_rescue_feed():
    assert override_target_labels("FlyHostel1_2X_2025-02-04_16-00-00__01") == OVERRIDE_TARGET_LABELS
    assert "feed" not in OVERRIDE_TARGET_LABELS
    assert override_target_labels("FlyHostel1_4X_2026-10-13_17-00-00_arena4__00") == ARENA_OVERRIDE_TARGET_LABELS
    assert "feed" in ARENA_OVERRIDE_TARGET_LABELS


def test_feed_counts_as_target_only_when_targeted():
    bouts = pd.DataFrame({"burst_id": [1, 1, 1, 1], "label": ["pe", "feed", "feed", "pe_near_food"], "dur_s": [1, 1, 1, 1]})
    assert label_composition(bouts)["frac_pe_or_target"].item() == 0.5
    assert label_composition(bouts, ARENA_OVERRIDE_TARGET_LABELS)["frac_pe_or_target"].item() == 1
