import argparse
from .proboscis_candidates import proboscis_candidates_for_fly
from .pe_features import pe_features_for_fly
from .extract_burst_traces import main as extract_burst_traces_for_fly
from .make_burst_clips import main as make_burst_clips_for_fly


def pipeline_for_fly(fly, n_jobs):
    proboscis_candidates_for_fly(fly)
    pe_features_for_fly(fly)
    extract_burst_traces_for_fly(fly, output = ".", n_jobs=n_jobs)
    make_burst_clips_for_fly(fly, upscale=1, output=".", n_jobs=n_jobs)
    

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fly", required=True)
    args=ap.parse_args()
    pipeline_for_fly(args.fly, n_jobs=1)


# ==========================================================================
if __name__ == "__main__":
    main()