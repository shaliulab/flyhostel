import sqlite3, io, numpy as np

def load_arena_mask_(dbfile):
    con = sqlite3.connect(dbfile)
    try:
        row = con.execute("SELECT mask FROM ROI_MAP LIMIT 1").fetchone()
    finally:
        con.close()
    return decode_mask_blob(bytes(row[0]))

def decode_mask_blob(b):
    if b[:6] == b'\x93NUMPY':                      # numpy .npy
        return np.load(io.BytesIO(b))
    try:                                            # PNG/JPEG
        import cv2
        a = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE)
        if a is not None: return a
    except Exception: pass
    try:
        from PIL import Image
        return np.array(Image.open(io.BytesIO(b)).convert('L'))
    except Exception: pass
    try:                                            # pickled array
        return np.asarray(np.load(io.BytesIO(b), allow_pickle=True))
    except Exception: pass
    raise ValueError(f"Unrecognized mask blob; first 16 bytes: {b[:16]!r}")

def arena_calib(dbfile, px_per_mm):
    m = load_arena_mask_(dbfile)
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