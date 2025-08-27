import numpy as np

def draw_partner_fly(frame, row, square_size=100, thickness=2, value=255):
    """
    Project the outline of an imaginary square (side = square_size) centered at
    (center_x_nn, center_y_nn) in ORIGINAL-image coordinates onto the current
    cropped `frame`, which is centered at (center_x, center_y) in ORIGINAL coords.

    Only the visible parts inside the crop are drawn, as an outline (no fill),
    with the given `thickness`.

    Parameters
    ----------
    frame : np.ndarray
        Cropped image (H,W) or (H,W,3). The crop comes from a larger original image.
    row : pd.Series
        Must contain:
          - 'center_x', 'center_y'  : original-coord center of this crop
          - 'center_x_nn','center_y_nn' : original-coord center of the imaginary square
    square_size : int
        Side length of the imaginary square in pixels (default 100).
    thickness : int
        Outline thickness in pixels (default 2).
    value : int or tuple
        Drawing value (e.g., 255 for grayscale, (255,255,255) for RGB).

    Returns
    -------
    frame : np.ndarray
        Modified `frame` with outline drawn (in place, and also returned).
    """
    H, W = frame.shape[:2]

    # Crop position in ORIGINAL coordinates
    crop_cx = float(row['center_x'])
    crop_cy = float(row['center_y'])

    # Target (imaginary frame) center in ORIGINAL coordinates
    x2 = float(row['center_x_nn'])
    y2 = float(row['center_y_nn'])

    # Original-space bbox of the crop
    crop_left   = crop_cx - W / 2.0
    crop_top    = crop_cy - H / 2.0
    crop_right  = crop_left + W
    crop_bottom = crop_top + H

    # Original-space bbox of the imaginary square
    half = square_size / 2.0
    sq_left, sq_right = x2 - half, x2 + half
    sq_top,  sq_bot   = y2 - half, y2 + half

    # Helpers
    def set_pixels(ys, xs):
        if frame.ndim == 2:
            frame[ys, xs] = value
        else:
            frame[ys, xs, ...] = value

    def thick_rows_around(y_float):
        start = int(np.floor(y_float - (thickness - 1) / 2.0))
        return range(start, start + thickness)

    def thick_cols_around(x_float):
        start = int(np.floor(x_float - (thickness - 1) / 2.0))
        return range(start, start + thickness)

    # Convert ORIGINAL x/y to CROP pixel coordinates
    def to_crop_x(x_orig): return x_orig - crop_left
    def to_crop_y(y_orig): return y_orig - crop_top

    # Draw horizontal edge at ORIGINAL y = y0 from x0..x1
    def draw_hline_orig(y0, x0, x1):
        # Map to crop coords
        y_crop_center = to_crop_y(y0)
        xa = int(np.ceil(max(0, min(to_crop_x(x0), to_crop_x(x1)))))
        xb = int(np.floor(min(W - 1, max(to_crop_x(x0), to_crop_x(x1)))))
        if xa > xb:
            return
        for y in thick_rows_around(y_crop_center):
            if 0 <= y < H:
                set_pixels(y, slice(xa, xb + 1))

    # Draw vertical edge at ORIGINAL x = x0 from y0..y1
    def draw_vline_orig(x0, y0, y1):
        x_crop_center = to_crop_x(x0)
        ya = int(np.ceil(max(0, min(to_crop_y(y0), to_crop_y(y1)))))
        yb = int(np.floor(min(H - 1, max(to_crop_y(y0), to_crop_y(y1)))))
        if ya > yb:
            return
        for x in thick_cols_around(x_crop_center):
            if 0 <= x < W:
                set_pixels(slice(ya, yb + 1), x)

    # Only draw if the (infinitely thin) line crosses the crop in ORIGINAL space
    def overlaps_hline(y0):
        return (crop_top - (thickness/2.0)) <= y0 <= (crop_bottom + (thickness/2.0))
    def overlaps_vline(x0):
        return (crop_left - (thickness/2.0)) <= x0 <= (crop_right + (thickness/2.0))

    # Outline edges (top, bottom, left, right) in ORIGINAL coordinates, projected to crop
    if overlaps_hline(sq_top):
        draw_hline_orig(sq_top,  sq_left, sq_right)
    if overlaps_hline(sq_bot):
        draw_hline_orig(sq_bot,  sq_left, sq_right)
    if overlaps_vline(sq_left):
        draw_vline_orig(sq_left, sq_top,  sq_bot)
    if overlaps_vline(sq_right):
        draw_vline_orig(sq_right, sq_top, sq_bot)

    return frame

# --- Example usage with your variables `frame` and `row` ---
# frame = draw_imaginary_frame_outline_on_crop(frame, row, square_size=100, thickness=2, value=255)


import numpy as np

def draw_partner_fly_translucent(
    frame, row, square_size=100, thickness=2, alpha=0.5, blend_to=0
):
    """
    Project the outline of an imaginary square (side = square_size) centered at
    (center_x_nn, center_y_nn) in ORIGINAL-image coordinates onto the current
    cropped `frame` (centered at center_x, center_y). Draw the outline only,
    blending translucently with the underlying pixels.

    Parameters
    ----------
    frame : np.ndarray
        Cropped image (H,W) or (H,W,3). Modifies in place and returns it.
    row : pd.Series
        Must contain:
          - 'center_x', 'center_y'            : original-coord center of this crop
          - 'center_x_nn','center_y_nn'       : original-coord center of the imaginary square
    square_size : int
        Side length of the imaginary square in pixels (default 100).
    thickness : int
        Outline thickness in pixels (default 2).
    alpha : float
        Blend weight toward `blend_to`. Output = (1 - alpha)*orig + alpha*blend_to.
        For a 50/50 average with black, use alpha=0.5 (default).
    blend_to : int or tuple
        Target color to blend toward. Use 0 or (0,0,0) for black (default).
        For grayscale, provide an int; for RGB, a 3-tuple.

    Returns
    -------
    frame : np.ndarray
        Modified frame with the translucent outline.
    """
    H, W = frame.shape[:2]

    # Crop center (original coords)
    crop_cx = float(row['center_x'])
    crop_cy = float(row['center_y'])

    # Imaginary square center (original coords)
    x2 = float(row['center_x_nn'])
    y2 = float(row['center_y_nn'])

    # Original-space bbox of the crop
    crop_left   = crop_cx - W / 2.0
    crop_top    = crop_cy - H / 2.0
    crop_right  = crop_left + W
    crop_bottom = crop_top + H

    # Original-space bbox of the imaginary square
    half = square_size / 2.0
    sq_left, sq_right = x2 - half, x2 + half
    sq_top,  sq_bot   = y2 - half, y2 + half

    # ----- Blending helpers -----
    def _dtype_max(arr):
        if np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            return float(info.max)
        # Assume float images are in 0..1 space
        return 1.0

    maxv = _dtype_max(frame)
    # Normalize to float for blending
    # We’ll write back in original dtype at the end of each draw call
    def blend_set(ys, xs):
        """Blend pixels at (ys, xs) toward blend_to by weight alpha."""
        if frame.ndim == 2:
            # Grayscale
            orig = frame[ys, xs].astype(np.float32) / maxv
            target = (float(blend_to) / maxv) if not isinstance(blend_to, tuple) else (float(blend_to[0]) / maxv)
            out = (1.0 - alpha) * orig + alpha * target
            frame[ys, xs] = np.clip(out * maxv, 0, maxv).astype(frame.dtype)
        else:
            # Color
            orig = frame[ys, xs, :].astype(np.float32) / maxv
            if isinstance(blend_to, tuple) or isinstance(blend_to, list):
                bt = np.asarray(blend_to, dtype=np.float32) / maxv
            else:
                bt = np.array([blend_to, blend_to, blend_to], dtype=np.float32) / maxv
            out = (1.0 - alpha) * orig + alpha * bt
            frame[ys, xs, :] = np.clip(out * maxv, 0, maxv).astype(frame.dtype)

    # Thickness helpers
    def thick_rows_around(y_float):
        start = int(np.floor(y_float - (thickness - 1) / 2.0))
        return range(start, start + thickness)

    def thick_cols_around(x_float):
        start = int(np.floor(x_float - (thickness - 1) / 2.0))
        return range(start, start + thickness)

    # Original -> crop coords
    def to_crop_x(x_orig): return x_orig - crop_left
    def to_crop_y(y_orig): return y_orig - crop_top

    # Draw horizontal edge (original y = y0) from x0..x1
    def draw_hline_orig(y0, x0, x1):
        y_c = to_crop_y(y0)
        x0c, x1c = to_crop_x(x0), to_crop_x(x1)
        xa = int(np.ceil(max(0, min(x0c, x1c))))
        xb = int(np.floor(min(W - 1, max(x0c, x1c))))
        if xa > xb:
            return
        for y in thick_rows_around(y_c):
            if 0 <= y < H:
                blend_set(y, slice(xa, xb + 1))

    # Draw vertical edge (original x = x0) from y0..y1
    def draw_vline_orig(x0, y0, y1):
        x_c = to_crop_x(x0)
        y0c, y1c = to_crop_y(y0), to_crop_y(y1)
        ya = int(np.ceil(max(0, min(y0c, y1c))))
        yb = int(np.floor(min(H - 1, max(y0c, y1c))))
        if ya > yb:
            return
        for x in thick_cols_around(x_c):
            if 0 <= x < W:
                blend_set(slice(ya, yb + 1), x)

    # Quick visibility checks (accounting for thickness)
    def overlaps_hline(y0):
        return (crop_top - (thickness/2.0)) <= y0 <= (crop_bottom + (thickness/2.0))
    def overlaps_vline(x0):
        return (crop_left - (thickness/2.0)) <= x0 <= (crop_right + (thickness/2.0))

    # Draw outline (top, bottom, left, right) as seen in the crop
    if overlaps_hline(sq_top):
        draw_hline_orig(sq_top,  sq_left, sq_right)
    if overlaps_hline(sq_bot):
        draw_hline_orig(sq_bot,  sq_left, sq_right)
    if overlaps_vline(sq_left):
        draw_vline_orig(sq_left, sq_top,  sq_bot)
    if overlaps_vline(sq_right):
        draw_vline_orig(sq_right, sq_top,  sq_bot)

    return frame

# --- Example ---
# frame = draw_partner_fly_translucent(frame, row, square_size=100, thickness=2, alpha=0.5, blend_to=0)
