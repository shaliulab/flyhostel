import numpy as np
from flyhostel.data.roi_map import load_roi_map, draw_roi_mask, decode_mask_blob

def load_arena_mask_(dbfile):
    """Arena mask (255 inside, 0 outside and in the negative ROIs)"""
    return load_roi_map(dbfile)["mask"]

def load_arena_rois_(dbfile):
    """List of (sign, contour) of the arena, sign is -1 for negative ROIs (areas excluded from the arena)"""
    return load_roi_map(dbfile)["rois"]

def arena_calib(dbfile, px_per_mm):
    roi_map = load_roi_map(dbfile)
    # the arena disk, without removing the negative ROIs (e.g. notches cut from its edge)
    m = draw_roi_mask([roi for roi in roi_map["rois"] if roi[0] > 0], roi_map["mask"].shape)
    ys, xs = np.nonzero(np.asarray(m) > (np.asarray(m).max() / 2))   # white = arena
    cx_cent, cy_cent = xs.mean(), ys.mean()                          # centroid
    cx_bb = (xs.min()+xs.max())/2.0; cy_bb = (ys.min()+ys.max())/2.0 # bbox center
    r_area = np.sqrt(xs.size/np.pi)                                  # from area
    r_ext  = np.sqrt((xs-cx_cent)**2 + (ys-cy_cent)**2).max()        # max extent
    # diagnostics: for a clean disk these should agree
    if np.hypot(cx_cent-cx_bb, cy_cent-cy_bb) > 0.02*r_area:
        print(f"WARN mask center: centroid {cx_cent:.1f},{cy_cent:.1f} vs bbox {cx_bb:.1f},{cy_bb:.1f} — not a clean disk?")
    if abs(r_area - r_ext) > 0.05*r_area:
        print(f"WARN mask radius: area {r_area:.1f} vs extent {r_ext:.1f} px — hole/notch/clipping?")
    r_mm = r_area / px_per_mm
    if not (27.0 <= r_mm <= 33.0):
        print(f"WARN arena radius {r_mm:.1f} mm != ~30 mm — check px_per_mm or mask")
    return cx_cent, cy_cent, r_area                                  # px, px, px


class ROILoader:

    def __init__(self, *args, **kwargs):
        self.dbfile=None
        super(ROILoader, self).__init__(*args, **kwargs)


    def load_arena_mask(self):
        return load_arena_mask_(self.dbfile)

    def load_arena_rois(self):
        return load_arena_rois_(self.dbfile)