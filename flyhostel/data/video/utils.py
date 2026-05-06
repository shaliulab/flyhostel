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
    
