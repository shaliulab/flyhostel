import glob
import itertools
import os
import sqlite3
import matplotlib.colors as colors
import cv2
import numpy as np
import joblib
from imgstore.interface import VideoCapture
from flyhostel.data.interactions.bodyparts import bodyparts, legs

animal_colors=["#1A281F", "#635255", "#CE7B91", "#C0E8F9", "#B8D3D1", "#F2E86D"]

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

def connect_bps(canvas, bp1, row, color, bp2=None):
    """
    Draw a point at the coordinates of a bodypart.
    If a second bodypart is provided,
    its point is also drawn, as well as the line connecting them
    """
    
    point_size=3
    line_size=1

    x0 = row["_".join([bp1, "x"])]
    y0 = row["_".join([bp1, "y"])]
    if ~np.isnan(x0):
        x0=int(x0)
        y0=int(y0)        
        canvas=cv2.circle(canvas, (x0, y0), point_size, color, -1)
        
    if bp2 is not None:
        x1 = row["_".join([bp2, "x"])]
        y1 = row["_".join([bp2, "y"])]

        if ~np.isnan(x1):
            x1=int(x1)
            y1=int(y1)
            canvas=cv2.circle(canvas, (x1, y1), point_size, color, -1)
        if ~np.isnan(x0) and ~np.isnan(x1):
            canvas=cv2.line(canvas, (x0, y0), (x1, y1), color, line_size)
    return canvas


def draw_fly(canvas, t, dt, id):
    """
    Draw the pose of an animal
    """
    dt_animal = dt.loc[id]

    row=dt_animal.iloc[np.abs(dt_animal["t"]-t).argmin()]
    identity = row["identity"].item()
    color = [int(255*e) for e in colors.hex2color(animal_colors[identity-1])]
    canvas=connect_bps(canvas, bp1="centroid", bp2="abdomen", row=row, color=color)
    canvas=connect_bps(canvas, bp1="head", bp2="proboscis", row=row, color=color)
    for leg in legs:
        canvas=connect_bps(canvas, bp1="centroid", bp2=leg, row=row, color=color)
  
    canvas=connect_bps(canvas, bp1="centroid", bp2="rightWing", row=row, color=color)
    canvas=connect_bps(canvas, bp1="centroid", bp2="leftWing", row=row, color=color)
    return canvas


def draw_frame(dbfile, dt, identities, t=None, frame_number=None, store=None):
    """
    Draw the pose of all animals at a given time
    """
    number_of_animals=len(identities)
    assert number_of_animals  <= len(animal_colors), f"Up to 6 animals supported. {number_of_animals} is too much"

    canvas = generate_canvas(dbfile)
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
        canvas=frame.copy()
    for id in range(number_of_animals):
        canvas =draw_fly(canvas, t, dt=dt, id=identities[id])
        # break
    return frame, canvas


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


def make_pose_segment(basedir, dt, output_folder=".", ts=None, frame_numbers=None, segment_number=0, roi=None):
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
            frame, canvas = draw_frame(dbfile, dt=dt, identities=identities, store=store, **{points_name: point})

            if roi is not None:
                x, y, w, h = roi
                canvas = canvas[y:(y+h), x:(x+w)]

            wh=canvas.shape[:2][::-1]
            
            if vw is None:
                path=output_folder+f"/video_with_pose_{str(segment_number).zfill(3)}.mp4"
                os.makedirs(output_folder, exist_ok=True)

                vw=cv2.VideoWriter(
                    path,
                    cv2.VideoWriter_fourcc(*"DIVX"),
                    fps=2,
                    frameSize=wh,
                    isColor=True,
                )
            vw.write(canvas)
    finally:
        if vw is not None:
            vw.release()
        store.release()

    return path