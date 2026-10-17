import os, cv2, numpy as np
from flyhostel.tests import synthetic_experiment as synthetic

def test_probe(experiment_copy, monkeypatch):
    monkeypatch.chdir(experiment_copy)
    from flyhostel.data.bin.video import main, get_parser
    args = get_parser().parse_args(["--experiment", synthetic.EXPERIMENT, "--dataset", synthetic.EXPERIMENT + ".db", "--n-jobs", "1", "--chunks", "1"])
    main(args)
    for root, dirs, files in os.walk("flyhostel"):
        for f in sorted(files):
            p = os.path.join(root, f)
            if f.endswith(".mp4"):
                cap = cv2.VideoCapture(p); print(p, cap.get(7), cap.get(3), cap.get(4))
            elif f.endswith(".txt"):
                print(p, open(p).read().strip())
            else:
                print(p)
    print(open("status.txt").read())
