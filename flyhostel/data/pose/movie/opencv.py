import logging
import webcolors
from tqdm.auto import tqdm
import cv2
from flyhostel.data.pose.constants import framerate as FRAMERATE

logger=logging.getLogger(__name__)

COLORS={
    "walk": "red",
    "groom": "green",
    "pe_inactive": "yellow",
    "pe_unknown": "yellow",
    "pe_hidden": "purple",
    "inactive": "purple",
    "feed": "orange",
}
COLORS={k: webcolors.name_to_rgb(v)[::-1] for k, v in COLORS.items()}
black=(0, 0, 0)

def adjust_text_size(image, text, desired_width_fraction, desired_height_fraction, base_font_scale=1.0, base_thickness=1):
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, base_font_scale, base_thickness)

    image_width, image_height = image.shape[1], image.shape[0]
    desired_width = image_width * desired_width_fraction
    desired_height = image_height * desired_height_fraction

    font_scale = base_font_scale * min(desired_width / text_width, desired_height / text_height)
    # thickness might need manual adjustment
    thickness = base_thickness
    return font_scale, thickness

def annotate_behavior_in_frame(frame, text, font_scale, thickness,x=0.7, y=0.1):

    frame_shape=frame.shape[:2]

    frame=cv2.resize(frame, (500, 500))
    tr = (int(frame.shape[1]*x), int(frame.shape[0]*y))
    frame=cv2.putText(
        frame,
        text,
        org=tr,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        color=COLORS.get(text, black),
        thickness=thickness
    )
    frame=cv2.resize(frame, frame_shape[::-1])

    return frame


def annotate_behavior_in_video_cv2(video_input, frame_idx, behaviors, video_output, gui_progress=False, fps=FRAMERATE, font_scale=None, thickness=None):
    vw=None
    cap=None

    try:
        cap=cv2.VideoCapture(video_input)
        i=0
        cap.set(1, frame_idx[i])
        logger.debug("Will save %s frames to ---> %s @ %s FPS", len(frame_idx), video_output, fps)
        if gui_progress:
            pb=tqdm(total=len(frame_idx))
        
        ret=True
        while ret:
            ret, frame=cap.read()
            frame=cv2.resize(frame, (500, 500))
            if font_scale is None:
                font_scale, thickness = adjust_text_size(frame, "behavior", 0.15, 0.15)
            if not ret:
                break
    
            frame=annotate_behavior_in_frame(frame, behaviors[i], font_scale=font_scale, thickness=thickness, x=0.7, y=0.1)
            if vw is None:
                logger.debug("Initializing ---> %s", video_output)
                vw=cv2.VideoWriter(
                    video_output,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps=fps,
                    frameSize=frame.shape[:2][::-1],
                    isColor=True
                )
                
            vw.write(frame)
            if gui_progress:
                pb.update(1)
            
    
            if i+1 == len(frame_idx):
                break
            
            if frame_idx[i+1]-frame_idx[i]>1:
                cap.set(1, frame_idx[i+1])

            i+=1

    except Exception as error:
        logger.error(error)
    
    finally:
        if cap is not None:
            cap.release()
        if vw is not None:
            vw.release()

