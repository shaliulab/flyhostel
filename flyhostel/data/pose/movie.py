import glob
import itertools
import logging

import os
import sqlite3
import matplotlib.colors as colors
import cv2
import numpy as np
import joblib
from imgstore.interface import VideoCapture
from flyhostel.data.interactions.bodyparts import bodyparts, legs

logger=logging.getLogger(__name__)
animal_colors=["#1A281F", "#635255", "#CE7B91", "#C0E8F9", "#B8D3D1", "#F2E86D"]
point_size=10
line_size=10
scale_factor=10

def generate_canvas(dbfile, color=(255, 255, 255)):
    """
    Generate a homogeneous background to draw in with a custom color
    """
    
    with sqlite3.connect(dbfile) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM METADATA where field in ('frame_width', 'frame_height');")
        [(frame_width,), (frame_height,)] = cursor.fetchall()
    frame_width=int(frame_width)
    frame_height=int(frame_height)
    
    canvas = np.zeros((frame_width, frame_height, 3), dtype=np.uint8)
    canvas[:]=color
    return canvas

def write_identity(canvas, row, identity, color):
    

    org = (int(row["centroid_x"]), int(row["centroid_y"]))
    # (label_width, label_height), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)


    canvas = cv2.putText(
        canvas,
        str(identity),
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
    )
    return canvas


def connect_bps(canvas, bp1, row, color, bp2=None):
    """
    Draw a point at the coordinates of a bodypart.
    If a second bodypart is provided,
    its point is also drawn, as well as the line connecting them

    Arguments:

        canvas (np.array): np.uint8 array on which to draw
        bp1 (str): Name of bodypart, a pair of columns called bp_x and bp_y should be present in row
        row (dict-like): Must contain entries bp_x and bp_y with an integer value which refers to the coordinates of the bp in the canvas
        color (tuple): 3 8-bit numbers encoding the RGB code of the point (and line color)
    """

    x1 = row["_".join([bp1, "x"])]
    y1 = row["_".join([bp1, "y"])]
    if ~np.isnan(x1):
        x1=int(x1)
        y1=int(y1)        
        canvas=cv2.circle(canvas, (x1, y1), point_size, color, -1)
        # canvas=cv2.putText(
        #     canvas, bp1, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     color,
        #     2
        # )
    else:
        logger.debug("bp1: %s not found", bp1)

        
    if bp2 is not None:
        x2 = row["_".join([bp2, "x"])]
        y2 = row["_".join([bp2, "y"])]

        if ~np.isnan(x2):
            x2=int(x2)
            y2=int(y2)
            canvas=cv2.circle(canvas, (x2, y2), point_size, color, -1)
        else:
            logger.debug("bp2: %s not found", bp2)

        if ~np.isnan(x1) and ~np.isnan(x2):
            canvas=cv2.line(canvas, (x1, y1), (x2, y2), color, line_size)
    return canvas


def draw_fly(canvas, t, dt, id, with_identity=False):
    """
    Draw the pose of an animal

    Arguments:

    canvas (np.array): np.uint8 array on which to draw
    t (int): Timestamp in dt from which data will be taken
    dt (pd.DataFrame): Must contain columns bp_x and bp_y where bp includes centroid head proboscis abdomen rightWing leftWing and legs
    id (str): Index value to filter draw_fly by
    """
    dt_animal = dt.loc[id]

    row=dt_animal.iloc[np.abs(dt_animal["t"]-t).argmin()]
    identity = row["identity"].item()
    color = [int(255*e) for e in colors.hex2color(animal_colors[identity-1])]
    canvas=connect_bps(canvas, bp1="centroid", bp2="abdomen", row=row, color=color)
    canvas=connect_bps(canvas, bp1="head", bp2="proboscis", row=row, color=color)
    canvas=connect_bps(canvas, bp1="head", bp2="centroid", row=row, color=color)
    for leg in legs:
        if "mid" in leg:
            canvas=connect_bps(canvas, bp1="centroid", bp2=leg, row=row, color=(0, 0, 0))
        else:
            canvas=connect_bps(canvas, bp1="centroid", bp2=leg, row=row, color=color)
  
    canvas=connect_bps(canvas, bp1="centroid", bp2="rightWing", row=row, color=color)
    canvas=connect_bps(canvas, bp1="centroid", bp2="leftWing", row=row, color=color)
    if with_identity:
        canvas = write_identity(canvas, row=row, identity=identity, color=color)
    return canvas


def draw_frame(dbfile, dt, identities, t=None, frame_number=None, store=None, with_identity=False, roi=None):
    """
    Draw the pose of all animals at a given time
    """
    number_of_animals=len(identities)
    assert number_of_animals  <= len(animal_colors), f"Up to 6 animals supported. {number_of_animals} is too much"

    white_canvas = generate_canvas(dbfile)
    if t is not None:
        row_number=np.abs(dt["t"]-t).argmin()
        frame_number = dt.iloc[row_number]["frame_number"].item()
    else:
        t = dt.loc[dt["frame_number"] == frame_number, "t"].values[0]
        row_number=np.abs(dt["t"]-t).argmin()

    chunk, frame_idx = store._index.find_chunk("frame_number", frame_number)
    if store is not None:
        store.set(1, frame_number)
        ret, frame = store.read()
        drawn_frame = frame.copy()

    canvas_shape = white_canvas.shape

    templates=[white_canvas]
    for i, canvas in enumerate(templates):

        canvas=cv2.resize(canvas, (canvas_shape[1]*scale_factor, canvas_shape[0]*scale_factor))
        for bp in bodyparts + ["centroid"]:
            for coord in ["x", "y"]:
                dt[f"{bp}_{coord}"]*=scale_factor

        for id in range(number_of_animals):
            canvas =draw_fly(canvas, t, dt=dt, id=identities[id], with_identity=with_identity)
            
        # cv2.imwrite("pose_movies/white_canvas.png", canvas)
        templates[i]=canvas

    white_canvas = templates[0]
    
    if roi is not None:
        x, y, w, h = roi
        inset_frame = drawn_frame[y:(y+h), x:(x+w)]
        inset_canvas = white_canvas[y*scale_factor:(y*scale_factor+h*scale_factor), x*scale_factor:(x*scale_factor+w*scale_factor)]
        frame=cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
        frame=cv2.resize(frame, (1000, 1000))

        img = np.hstack([
            frame,
            np.vstack([cv2.resize(inset_frame, (500, 500)), cv2.resize(inset_canvas, (500, 500))])
        ])
    else:
        frame=cv2.resize(frame, (1000, 1000))
        img = frame
        
    return img


def partition_list(input_list, chunk_size):
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i:i + chunk_size]


def make_pose_movie(basedir, dt, ts=None, frame_numbers=None, n_jobs=24, **kwargs):
    if ts is None:
        points_name="frame_numbers"
        points = frame_numbers
    else:
        points_name="ts"
        points = ts
    
    points_partition = list(partition_list(points, 300))
    
    return joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(
            make_pose_segment
        )(
            basedir, dt, **{points_name: points}, segment_number=i, **kwargs
        )
        for i, points in enumerate(points_partition)
    )


def make_pose_segment(basedir, dt, output_folder=".", ts=None, frame_numbers=None, segment_number=0, roi=None, fps=2, with_identity=False):
    f"""
    dt:
    index is id
    must contain column t with time (should be seconds since zt0 but not required)
    must contain column frame_number with the corresponding frame number of the original recording
    must contain two columns called bp_x and bp_y where bp is all of the bodyparts in {bodyparts}. Must also contain centroid_x and centroid_y 

    roi = x, y, w, h
    """

    columns=[[bp + "_x", bp + "_y"] for bp in bodyparts + ["centroid"]]
    columns=itertools.chain(*columns)
    assert [col in dt.columns for col in columns]

    dbfiles = glob.glob(basedir + "/FlyHostel*.db")
    assert len(dbfiles) == 1, f"{len(dbfiles)} FlyHostel dbfiles found, but only one should be present"
    dbfile = dbfiles[0]

    try:
        store=VideoCapture(basedir + "/metadata.yaml", 100)
        vw=None
        if ts is None:
            assert frame_numbers
            points_name = "frame_number"
            points=frame_numbers
            points=[int(e) for e in points]
            
        else:
            points_name = "t"
            points = ts
            points=[float(e) for e in points]
    

        identities=sorted(np.unique(dt.index))
            
        for point in points:
            img = draw_frame(
                dbfile, dt=dt.copy(), identities=identities, store=store, **{points_name: point},
                with_identity=with_identity,
                roi=roi
            )
            
            wh=img.shape[:2][::-1]
            if vw is None:
                path=output_folder+f"/video_with_pose_{str(segment_number).zfill(3)}.mp4"
                os.makedirs(output_folder, exist_ok=True)
                vw=cv2.VideoWriter(
                    path,
                    cv2.VideoWriter_fourcc(*"DIVX"),
                    fps=fps,
                    frameSize=wh,
                    isColor=True,
                )
            vw.write(img)
    finally:
        if vw is not None:
            vw.release()
        store.release()

    return path