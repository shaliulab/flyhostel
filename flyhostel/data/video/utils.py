from flyhostel import lab_paths
import cv2
import logging
logger=logging.getLogger(__name__)

# def add_info_box(frame, properties, series, font_scale=1):
#     """
#     Adds an information box in the top-right corner of the frame.
    
#     Parameters:
#         frame (np.array): The image/frame to modify.
#         properties (list): List of property names to display.
#         series (pd.Series): Pandas Series containing values for the properties.
    
#     Returns:
#         np.array: The modified frame with the information box.
#     """
#     # Define box dimensions based on text size
#     box_width = 500  # Fixed width for consistency
#     thickness = 1
#     padding = 10  # Padding inside the box
#     line_spacing = 30  # Space between lines

#     # Determine total height needed
#     box_height = padding * 2 + line_spacing * len(properties)

#     # Define the top-right corner box area
#     img_h, img_w, _ = frame.shape
#     x1, y1 = img_w - box_width - 10, 10  # Small margin from top-right corner
#     x2, y2 = x1 + box_width, y1 + box_height

#     # Draw white rectangle background
#     cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)

#     # Overlay text (black on white)
#     y_text = y1 + padding + 15  # Start below padding
#     for prop in properties:
#         value = str(series.get(prop, "N/A"))  # Get value from Series, default to "N/A"
#         text = f"{prop}: {value}"
#         cv2.putText(frame, text, (x1 + padding, y_text), cv2.FONT_HERSHEY_SIMPLEX,
#                     font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
#         y_text += line_spacing  # Move to next line

#     return frame

import cv2
import numpy as np
import pandas as pd

def add_info_box_gpt(frame, properties, series, font_scale=0.5):
    """
    Adds a visually aligned information box in the top-right corner of the frame.
    
    Parameters:
        frame (np.array): The image/frame to modify.
        properties (list): List of property names to display.
        series (pd.Series): Pandas Series containing values for the properties.
    
    Returns:
        np.array: The modified frame with the information box.
    """
    # Copy frame to avoid in-place drawing
    overlay = frame.copy()

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 1
    font_color = (0, 0, 0)  # black
    bg_color = (255, 255, 255)  # white
    alpha = 0.85  # transparency of the box

    # Padding and spacing
    padding = 10
    line_spacing = 5  # gap between lines

    # Measure max text width
    text_sizes = []
    lines = []
    for prop in properties:
        value = str(series.get(prop, "N/A"))
        text = f"{prop:<5}: {value:>11}"
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_sizes.append(text_size)
        lines.append(text)

    # Determine box size
    box_width = max(w for (w, h) in text_sizes) + 2 * padding
    line_height = max(h for (w, h) in text_sizes)
    box_height = len(lines) * (line_height + line_spacing) + 2 * padding - line_spacing

    # Position box in top-right corner
    img_h, img_w, _ = frame.shape
    x1, y1 = img_w - box_width - 10, 10
    x2, y2 = x1 + box_width, y1 + box_height

    # Draw background box with transparency
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Draw right-aligned text
    y_text = y1 + padding + line_height
    for line, (w, h) in zip(lines, text_sizes):
        x_text = x2 - padding - w  # right-aligned: end of text aligns with box's right edge
        cv2.putText(frame, line, (x_text, y_text), font, font_scale,
                    font_color, font_thickness, cv2.LINE_AA)
        y_text += line_height + line_spacing


    return frame


def draw_sleep_state(img, data, org, radius, color, step_count):
    """
    Annotate sleep state of animal

    Arguments
        img (np.array)
        data (pd.Series)
        org (tuple)
    """
    if step_count==0:
        if "asleep" in data.index:
            if data["asleep"]==True:
                darker_color = tuple(max(0, int(c * 0.6)) for c in color)  # 60% brightness
                contour_color=(132, 210, 246)
                contour_thickness = max(2, int(radius * 0.1))
                img = cv2.circle(img, org, radius + contour_thickness, contour_color, contour_thickness)  # Larger for border
        else:
            logger.warning("asleep information not provided. Ignoring")
    return img
    


# ---------------------------------------------------------------------------------------------------------------------------------
# Time label for videos meant for talks and papers: the ZT in a rounded translucent panel, over a light-dark bar of the day (ZT0-12
# light, ZT12-24 dark) with a marker at the time of the frame. It replaces the frame number / ZT box (add_info_box_gpt).
# ---------------------------------------------------------------------------------------------------------------------------------
import functools
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

TIME_LABEL_FONTS={
    "regular": [lab_paths.LIBERATION_SANS, lab_paths.DEJAVU_SANS],
    "bold": [lab_paths.LIBERATION_SANS_BOLD, lab_paths.DEJAVU_SANS_BOLD],
}
TIME_LABEL_MARKER=(230, 159, 0)  # RGB, amber (Okabe-Ito orange)


@functools.lru_cache(maxsize=None)
def _time_label_font(weight, size):
    for path in TIME_LABEL_FONTS[weight]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def add_time_label(frame, t, caption=None, corner="top-right", scale=None, light_hours=12):
    """
    Draws the ZT of the frame in a corner: "ZT 14:32" over a 24 h light-dark bar with a marker at that time, and an optional caption
    under it (e.g. "360x"). Sizes follow the width of the frame (scale 1 = 2100 px, the flyhostel recordings).

    frame (np.array): H x W x 3, in the colour order of the caller (the panel is white, grey and black, the marker amber in RGB)
    t (float): seconds since ZT0 (may exceed 24 h; the clock wraps)
    """
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return frame
    h, w=frame.shape[:2]
    s=w/2100 if scale is None else scale
    zt=(float(t)/3600) % 24
    text=f"ZT {int(zt):02d}:{int(round((zt % 1)*60)) % 60:02d}"

    font_big=_time_label_font("bold", max(int(56*s), 10))
    font_small=_time_label_font("regular", max(int(26*s), 8))
    pad=int(22*s)
    margin=int(28*s)
    bar_w, bar_h=int(320*s), int(16*s)
    gap=int(14*s)
    gap_bar=int(30*s)  # room for the marker's triangle between the ZT and the bar

    text_box=font_big.getbbox(text)
    text_h=text_box[3]
    tick_h=font_small.getbbox("24")[3]
    caption_h=font_small.getbbox(caption)[3] if caption else 0
    panel_w=max(bar_w, text_box[2], font_small.getbbox(caption)[2] if caption else 0)+2*pad
    panel_h=pad+text_h+gap_bar+bar_h+int(6*s)+tick_h+(gap+caption_h if caption else 0)+pad

    x0=w-panel_w-margin if corner.endswith("right") else margin
    y0=margin if corner.startswith("top") else h-panel_h-margin

    image=Image.fromarray(frame).convert("RGBA")
    overlay=Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw=ImageDraw.Draw(overlay)
    draw.rounded_rectangle((x0, y0, x0+panel_w, y0+panel_h), radius=int(18*s), fill=(20, 20, 20, 170))

    # ZT
    x=x0+pad
    y=y0+pad
    draw.text((x, y), text, font=font_big, fill=(255, 255, 255, 255))
    y+=text_h+gap_bar

    # light-dark bar: light from ZT0 to light_hours, dark after; white outline
    light_w=int(bar_w*light_hours/24)
    draw.rectangle((x, y, x+light_w, y+bar_h), fill=(245, 245, 245, 255))
    draw.rectangle((x+light_w, y, x+bar_w, y+bar_h), fill=(0, 0, 0, 255))
    draw.rectangle((x, y, x+bar_w, y+bar_h), outline=(255, 255, 255, 255), width=max(int(2*s), 1))
    # marker at the current ZT: a line across the bar and a triangle above it
    mx=x+int(bar_w*zt/24)
    half=int(9*s)
    draw.line((mx, y-int(3*s), mx, y+bar_h+int(3*s)), fill=TIME_LABEL_MARKER+(255,), width=max(int(4*s), 2))
    draw.polygon([(mx-half, y-int(3*s)-half*1.4), (mx+half, y-int(3*s)-half*1.4), (mx, y-int(3*s))], fill=TIME_LABEL_MARKER+(255,))
    # ticks 0, 12, 24
    y_ticks=y+bar_h+int(6*s)
    for zt_tick in (0, 12, 24):
        label=str(zt_tick)
        lw=font_small.getbbox(label)[2]
        tx=x+int(bar_w*zt_tick/24)-lw//2
        tx=min(max(tx, x), x+bar_w-lw)
        draw.text((tx, y_ticks), label, font=font_small, fill=(210, 210, 210, 255))
    y=y_ticks+tick_h

    if caption:
        draw.text((x, y+gap), caption, font=font_small, fill=(210, 210, 210, 255))

    return np.array(Image.alpha_composite(image, overlay).convert("RGB"))
