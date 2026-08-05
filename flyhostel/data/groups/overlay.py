import numpy as np, h5py, cv2, matplotlib.pyplot as plt
from flyllm.config import keypointnames, skeleton_edges
from flyllm.features import kp2feat, feat2kp, compute_scale_perfly, body_centric_kp
from flyhostel.data.pose.loaders.roi import arena_calib
from flyhostel.data.groups.apf import read_fly

from .synthesize import FLIP_Y

SYNTH_KPTS = {'left_eye','right_eye','left_front_thorax','right_front_thorax',
              'left_middle_femur_base','right_middle_femur_base'}


def bodyframe_x(X_single):
    """Body-frame x-coordinate per keypoint for a single-frame (19,2,1,1) array.
    Negative = APF-left side, positive = APF-right (per kp2feat's pi-arctan2 convention).
    """
    Xn = np.asarray(body_centric_kp(X_single[:, :, :, 0])[0])   # (19,2,T,1)
    return Xn[:, 0, 0, 0]                                        # x, t=0, fly=0 -> (19,)


def overlay_apf_roundtrip(loader, h5_path, keep_map, npz_frame, mp4_path, video_frame,
                          scale=None, transform=True, global_video=True):
    px_per_mm = loader.px_per_mm
    cx, cy, _ = arena_calib(loader.dbfile, px_per_mm)
    raw_i = int(keep_map[npz_frame])

    # rebuild the full 19-kpt APF input for this fly, take a tiny window around the frame
    Xfull = read_fly(h5_path, (cx, cy), px_per_mm)              # (19,2,T) mm, APF frame
    f = npz_frame
    X = Xfull[:, :, max(0, f-1):f+2, None].astype(np.float64)   # (19,2,~3,1)
    X = X[:, :, ~np.all(np.isnan(X), axis=(0, 1, 3))]
    Tt = X.shape[2]
    flyid = np.zeros((Tt, 1), int)
    if scale is None:
        scale = compute_scale_perfly(X)
    ti = min(1, Tt - 1)                                         # frame index within the window

    # round-trip through APF's representation (the thing under test)
    if transform:
        Xrt = feat2kp(kp2feat(X, scale, flyid=flyid), scale, flyid=flyid)   # (19,2,Tt,1)
    else:
        Xrt = X.copy()
    xy_mm = Xrt[:, :, ti, 0].T                                  # (2,19) mm frame

    # invert read_fly's geometry: mm -> absolute px -> (crop px)
    with h5py.File(h5_path, "r") as fh:
        anchor = fh["anchor"][raw_i].astype(np.float64)
    inv = xy_mm * px_per_mm
    if FLIP_Y:
        inv[1] = -inv[1]
    inv[0] += cx
    inv[1] += cy
    if not global_video:          # crop-space image -> subtract anchor; else absolute-space
        inv[0] -= anchor[0]
        inv[1] -= anchor[1]

    # draw skeleton on the frame
    cap = cv2.VideoCapture(mp4_path); cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame)
    ok, img = cap.read(); cap.release(); assert ok
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(6, 6)); ax.imshow(img)
    for a, b in skeleton_edges:
        ax.plot([inv[0, a], inv[0, b]], [inv[1, a], inv[1, b]], '-', color='lime', lw=0.8)
    for k, nm in enumerate(keypointnames):
        c = 'red' if nm.startswith('left') else 'blue' if nm.startswith('right') else 'yellow'
        ax.plot(inv[0, k], inv[1, k], 'o', color=c, ms=4)
    ax.set_title(("feat2kp(kp2feat(X))" if transform else "X") +
                 " on raw crop • red=APF-left blue=APF-right")
    plt.savefig("overlay_roundtrip.png", dpi=140); print("wrote overlay_roundtrip.png")

    # --- handedness diagnostic: body-frame x-sign, raw X vs round-tripped Xrt ---
    Xraw_1 = X[:, :, ti:ti+1, :]                                # (19,2,1,1)
    Xrt_1  = Xrt[:, :, ti:ti+1, :]
    xr  = bodyframe_x(Xraw_1)
    xrt = bodyframe_x(Xrt_1)
    print("shapes:", xr.shape, xrt.shape)                      # expect (19,) (19,)
    print(f"{'keypoint':<32}{'raw_x':>7}{'rt_x':>7}  type")
    for k, nm in enumerate(keypointnames):
        a, b = float(xr[k]), float(xrt[k])
        flag = "  <-- SWAPPED" if np.sign(a) != np.sign(b) and abs(a) > 0.02 else ""
        tag  = "SYNTH" if nm in SYNTH_KPTS else "track"
        print(f"{nm:<32}{a:+7.3f}{b:+7.3f}  {tag}{flag}")