# roi-proposer

Proposes `DATE_TIME.toml` (arena ellipse + idtrackerai settings) and `landmarks.toml` (food ellipse + 4 notch polygons)
for a new experiment, so they only need to be checked and adjusted.

```bash
conda activate roi-proposer
roi-proposer propose /flyhostel_data/videos/FlyHostel1/3X/2025-06-21_15-00-00 [more experiments...]
```

What gets written depends on the folder:

| folder | writes | ROIs |
|---|---|---|
| `FlyHostelN/NX/DATE_TIME` | `DATE_TIME.toml` + `landmarks.toml` | arena, food, 4 notches: YOLO segmentation model trained on all experiments that already have both tomls |
| `FlyHostelN/NX/DATE_TIME` with several arenas (auto-detected, or `--arenas`) | `arenas.toml` | one ellipse per arena (no ML: Hough circles + floor edge), 5% larger than the floor like the hand-made ones |
| `FlyHostelN/NX/DATE_TIME/arenas/K` | `DATE_TIME.toml` + identical `landmarks.toml` | the arena floor, detected in the original recording and shifted with the crop offset in `metadata.yaml` |

* The image is the snapshot of chunk 100 (or the last chunk, or `--chunk`), or the first frame of the chunk `.mp4` if there is no `.png` yet.
* `intensity_ths`, `area_ths`, `track_wo_identities`...: copied from the most recent annotated experiment of the same kind
  with the same camera (preferring other arenas of the same experiment, then the same hostel).
* `number_of_animals` comes from the `NX` folder (1 in `arenas/K` folders, 0 in `arenas.toml`), `--number-of-animals` overrides it.
* A preview `rois_proposal.jpg` is saved in the experiment folder. Existing tomls are not overwritten without `--force`.
  `--dry-run` prints the tomls instead of writing them. Missing ROIs (e.g. food not visible) are reported as warnings.

Accuracy vs the hand-made ROIs: on the 60 most recent experiments (not used for training) the median center error is
12 px for the arena, 3 px for the food and 6 px for the notches (2100 px images); arena experiments are within 13 px and 5% in size.

## Retraining (e.g. once more experiments are annotated)

```bash
roi-proposer build-dataset   # data/ : copies of the snapshots + labels rendered from the tomls, latest 60 experiments = validation
roi-proposer train           # runs/roi_proposer/weights/best.pt
roi-proposer evaluate        # error vs hand-made ROIs on the validation experiments, previews in runs/eval/
```

`data/` and `runs/` live in `~/.local/share/flyhostel/roi_proposer` (override with `FLYHOSTEL_ROI_PROPOSER_DIR`).

## Installation

ultralytics needs opencv>=4 while flyhostel pins opencv<4, so it gets its own environment and flyhostel is installed without its dependencies:

```bash
mamba create -n roi-proposer python=3.11 && conda activate roi-proposer
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126  # driver supports CUDA <= 12.8
pip install ultralytics==8.3.40 "numpy<2" opencv-python-headless==4.10.0.84 tomli pyyaml confapp-shaliulab
pip uninstall -y opencv-python  # pulled in by ultralytics, clashes with the headless build
pip install --no-deps -e /path/to/flyhostel
```
