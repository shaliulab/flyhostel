import json
import os.path
import tempfile

import cv2
import pandas as pd
import numpy as np
import plotly.io
import plotly.graph_objs as go
import joblib

from flyhostel.data.pose.constants import framerate

SIZE=640
ROOT_DIR="/opt/behavior-viewer/static/datasets"

def thread(video_url, fig, index, scenes, i):
    scene=scenes[i]
    index_row=index.iloc[i]
    filename=index_row["animal"] + "__" + str(index_row["frame_number"]).zfill(10) + ".mp4"
    write_scene(video_url, fig, scene, filename="movies/" + filename)
    
    
def read_frame(cap, size):
    ret, image = cap.read()
    if ret:
        image=cv2.resize(image, (size, size))
        # image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return ret, image
    else:
        return False, None


def update_fig(fig, center_x, t1, t2, png_file):
    fig.update_layout(
        xaxis=dict(
            range=(t1, t2)
        ),
        shapes=[
            go.layout.Shape(
                type="line",
                x0=center_x,
                y0=0,  # Assuming your y-axis starts at 0
                x1=center_x,
                y1=1,  # Assuming your y-axis ends at 1
                xref="x",
                yref="paper",  # 'paper' refers to the entire range of the y-axis
                line=dict(
                    color="Black",
                    width=3
                )
            )
        ]
    )
    fig.write_image(png_file.name)
    plot=cv2.imread(png_file.name)
    return plot

def write_scene(url, fig, scene, filename, resolution=1, margin=10):
    """
    Make video of a behavior_viewer-like interface

    Arguments:
        url (str): Path to movie.m3u8 of one animal
        fig (plotly.graph_objs.Figure): instance containing the ethogram of the same animal. Generated using flyhostel.data.pose.ethogram.draw_ethogram
        scene (tuple): First and last frame number that the video will display
        filename (str): Path to output video
        resolution (int): Temporal resolution of the ethogram in fig (how many seconds are represented by each ethogram bin)
        margin (int): How many more seconds in the future and in the past around the scene to display in the ethogram (even if the video does not play them).
            Useful to show the context of what is played in the video

    Returns:
        None

    """
    vw=None
    cap=cv2.VideoCapture(url)
    cap.set(1, scene[0])
    png_file=tempfile.NamedTemporaryFile(suffix=".png", prefix="behavior-viewer")
    step_size=framerate*resolution

    for frame_number in range(scene[0], scene[1], step_size):
        print(frame_number)
        
        t1=scene[0]/framerate - margin
        t2=scene[1]/framerate + margin
        center_x=frame_number/framerate

        plot=update_fig(fig, center_x, t1, t2, png_file)

        # plot=cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)
        newsize=None
        
        for i in range(step_size):
            ret, image=read_frame(cap, SIZE)
            if not ret or image is None:
                raise ValueError(f"Cannot read frame {frame_number+i} of {url}")
                
                
            padding=np.ones((image.shape[0], 400, 3), dtype=np.uint8)*255
    
            image=np.hstack([padding, image, padding])
            # plot=cv2.resize(plot, (640, 700))
            if newsize is None:
                newsize=(image.shape[1], int(plot.shape[0]*image.shape[1]/plot.shape[1]))
                plot=cv2.resize(plot, newsize)
            
            img=np.vstack([
                image,
                plot
            ])
            
            if vw is None:
                vw=cv2.VideoWriter(
                    filename,
                    cv2.VideoWriter_fourcc(*"MP4V"),
                    frameSize=img.shape[:2][::-1],
                    fps=framerate,
                    isColor=True
                    
                )
            vw.write(img)

    vw.release()

def make_videos_of_extremely_brief_grooming_bouts(animal, index, n_jobs=20):

    video_url=os.path.join(ROOT_DIR, animal, "movie", "movie.m3u8")
    data=pd.read_feather(os.path.join(ROOT_DIR, animal, f"{animal}.feather"))
    fn0=data["frame_number"].iloc[0]
    
    with open(os.path.join(ROOT_DIR, animal, f"{animal}.json")) as handle:
        json_str=json.dumps(json.load(handle))
    fig = plotly.io.from_json(json_str)
        
    index_single_animal=index.loc[index["animal"]==animal]
    
    scenes=[]
    for _, row in index_single_animal.iterrows():
        
        fn1=int(row["frame_number"]-fn0 -5*framerate)
        # t1=fn1/framerate
        fn2=int(row["frame_number"]-fn0 +(5+row["duration"])*framerate)
        
        # t2=fn2/framerate
        # center_x=round((t1+t2)/2)
        scenes.append((fn1, fn2))

    joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(
            thread
        )(
            video_url, fig, index_single_animal, scenes, i
        )
        for i in range(len(scenes))
    )