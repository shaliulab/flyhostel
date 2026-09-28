import argparse
from .proboscis_candidates import proboscis_candidates_for_fly
from .pe_features import pe_features_for_fly
from .extract_burst_traces import main as extract_burst_traces_for_fly
from .make_burst_clips import main as make_burst_clips_for_fly
from .label_overrides import write_overrides


def override_labels_with_cv_and_mean_duration_criteria(fly, output="."):
    """Rescue pe_near_food bouts in regular, PE-paced bursts as PE.

    Writes pe_bouts/{fly}_pe_bouts.overrides.csv next to the feather; the feather
    itself is not modified. Must run after extract_burst_traces (needs the traces).
    The rule and its thresholds live in label_overrides.py.
    """
    return write_overrides(fly, output=output)

def pipeline_for_fly(fly, n_jobs):
    proboscis_candidates_for_fly(fly)
    pe_features_for_fly(fly)
    extract_burst_traces_for_fly(fly, output = ".", n_jobs=n_jobs)
    make_burst_clips_for_fly(fly, upscale=1, output=".", n_jobs=n_jobs)
    write_overrides(fly, output = ".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fly", required=True)
    args=ap.parse_args()
    pipeline_for_fly(args.fly, n_jobs=1)


# ==========================================================================
if __name__ == "__main__":
    main()