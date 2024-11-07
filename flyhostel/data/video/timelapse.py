"""
Generate timelapse videos from a flyhostel experiment
for visualization purposes
"""

import os
import logging
import yaml
from imgstore.interface import VideoCapture
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import cv2
import joblib
import numpy as np
import joblib
from tqdm import tqdm
from flyhostel.utils import (
    load_roi_width,
    get_dbfile
)
from flyhostel.data.pose.constants import (
    framerate,
    chunksize
)
plt.set_cmap("gray")
from matplotlib import cm
logger=logging.getLogger(__name__)

MISSING_DATA=(0, 0)

def get_spaced_colors_util(n, norm=False, black=True, cmap="jet"):
    RGB_tuples = cm.get_cmap(cmap)
    if norm:
        colors = [RGB_tuples(i / n) for i in range(n)]
    else:
        RGB_array = np.asarray([RGB_tuples(i / n) for i in range(n)])
        BRG_array = np.zeros(RGB_array.shape)
        BRG_array[:, 0] = RGB_array[:, 2]
        BRG_array[:, 1] = RGB_array[:, 1]
        BRG_array[:, 2] = RGB_array[:, 0]
        colors = [tuple(BRG_array[i, :] * 256) for i in range(n)]
    if black:
        black = (0.0, 0.0, 0.0)
        colors.insert(0, black)
    return colors


def draw_trace(df, img, fn, identities, step, n_steps, roi_width=None):
    """
    step: How many frames between each point that makes up the trace
    n_steps: How many steps in the trace

    The last point in the trace will be n_steps*step frames back in time
    """
    colors=get_spaced_colors_util(len(identities), black=False)

    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for j, identity in enumerate(identities):
        dff=df.loc[df["identity"]==identity]
        last_x = None
        last_y = None
        for i in range(n_steps):
            fn0 = fn
            fn1 = fn - step*i
            row=dff.loc[dff["frame_number"]==fn1]
            if row.shape[0] == 0:
                continue
            size = n_steps-i
            size*=1.5

            radius=int(size/2)
            thickness=int(size//2)
            thickness=max(thickness, 1)
            # print(id, i, fn1, radius)
            if len(row)==0:
                logger.warning("Identity %s not found in frame %s", identity, fn1)
                continue

            try:
                x_r = row["x"].item()
                y_r = row["y"].item()
            except:
                print(row)
                continue
            try:
                if np.isnan(x_r) or np.isnan(y_r):
                    logger.warning("Identity %s not found in frame %s", identity, fn1)
                    continue
            except:
                import ipdb; ipdb.set_trace()

            if roi_width is None:
                x=x_r
                y=y_r
            else:
                x = int(x_r * roi_width)
                y = int(y_r * roi_width)

            org = (x, y)
            if org != MISSING_DATA:
                img=cv2.circle(img, org, radius, colors[j], -1)
                last_org= (last_x, last_y)
                if last_x is not None and last_org!=MISSING_DATA:
                    img = cv2.line(img, last_org, org, colors[j], thickness)

            last_x = x
            last_y = y

    return img


def draw_frame(df, t_index, basedir, fns, identities, **kwargs):
    """
    df: Contains columns x, y (position in arena referred to the top left corner and normalized to the roi width) frame_number
    t_index: Contains columns frame_number, t (ZT time in seconds)
    """
    if len(fns)==0:
        logger.warning("draw_frame passed no frame numbers")
        return [(None, None)]
    store=VideoCapture(os.path.join(basedir, "metadata.yaml"), int(fns[0]//chunksize))
    imgs=[]
    for fn in fns:
        t=t_index.loc[t_index["frame_number"]==fn, "t"].values[0]
        try:
            plt.show()
            store.set(1, int(fn))
            ret, frame = store.read()
            assert ret, f"Cannot read frame {fn} from {basedir}"
            img = draw_trace(df, frame, fn, identities, **kwargs)
        except Exception as error:
            img=None
            logger.error(error)
            #raise error
        imgs.append((img, t))
    store.release()
    return imgs


def setup_background():
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal', 'box')
    ax.axis('off')  # Turn off the axis

    # Draw the top white half-circle
    top_circle = patches.Wedge(center=(0, 0), r=1, theta1=0, theta2=180, color='#EEEEEE')
    ax.add_patch(top_circle)

    # Draw the bottom black half-circle
    bottom_circle = patches.Wedge(center=(0, 0), r=1, theta1=180, theta2=360, color='black')
    ax.add_patch(bottom_circle)

    # Draw a blue border around the clock
    border_circle = patches.Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=5)
    ax.add_patch(border_circle)
    return fig, ax


tip_circle = None
scaling_factor = 0.9


def draw_clock(hour, clock_size=200):
    global tip_circle

    fig, ax=setup_background()

    # Initial dummy data for hour hand line and tip circle
    hour_hand_line, = ax.plot([0, 0], [0, 0], color='red', linewidth=5)
    tip_circle = patches.Circle((0, 0), .1, fill=True, color='red', zorder=3)
    ax.add_patch(tip_circle)

    # Convert hours to radians: 0 hours corresponds to 3 o'clock position
    # We subtract 90 degrees (or pi/2 radians) to start at the 12 o'clock position.
    radian = np.radians(15 * hour - 90)

    # Calculate end points of the hour hand
    end_x = np.sin(radian) * scaling_factor
    end_y = np.cos(radian) * scaling_factor

    # Update hour hand line data and tip circle position
    hour_hand_line.set_data([0, end_x], [0, end_y])
    tip_circle.set_center((end_x, end_y))

    # Render the figure to a canvas
    fig.canvas.draw()

    # Convert canvas data to a numpy array
    clock = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    clock = clock.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    quarter_size = clock.shape[0]//4
    s=0
    clock = clock[(
        quarter_size-s):(quarter_size*3)+s*2,
        quarter_size-s:(quarter_size*3)+s*2
    ]

    clock = cv2.resize(clock, (clock_size, clock_size))

    plt.close(fig)  # Close the figure
    return clock


def save_video(imgs_t, filename, mask, fps, clock_size=None):
    video_writer = None
    for i, (img, t) in enumerate(imgs_t):
        img=img.copy()
        if clock_size is not None:
            clock = draw_clock(t/3600, clock_size)
            top_left_y=0
            top_left_x = img.shape[1]-clock_size
            for c in range(3):
                img[
                    top_left_y:top_left_y+clock.shape[0],
                    top_left_x:top_left_x+clock.shape[1], c
                ][mask] = clock[:, :, c][mask]

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        wh = img.shape[:2][::-1]
        if video_writer is None:
            video_writer=cv2.VideoWriter(
                filename,
                cv2.VideoWriter_fourcc(*"DIVX"),
                fps,
                wh,
                isColor=True,
            )

        video_writer.write(img)
    video_writer.release()


def make_timelapse_from_data(
    df,
    output_video,
    basedir,
    identities=None,
    roi_width=None,
    back_in_time=60*15,
    seconds_between=60,
    fps=15,
    min_chunk=0,
    max_chunk=float("inf"),
    n_jobs=20,
    partition_size=5,
    framerate=150,
):
    """
    df (pd.DataFrame): contains columns frame_number, t, x, y, identity
        x and y must be normalized coordinates in the arena, from 0 (left) to 1 (right)
        t must be seconds since zt0
        frame_number
        identity must be an integer starting from 1 that is unique and specific to each animal 
    output_video (str): path to the produced video
    basedir (str): path to the imgstore that produced the dataset in df
    seconds_between (int): seconds passed between each point in the trace
    back_in_time (int): how many seconds back in time should the traces look back to.
        must be multiple of back_in_time, it controls how long the traces are
    fps (int): frames per second of the output.
      (how many frames are displayed in the resulting video, per second)
    framerate (int): frames per second of the original recording
        i.e. during one second of experiment, how many frames were collected by the camera?
    partition_size (int): how many frames will each job process before being restarted 
    """

    if identities is None:
        identities=df["identity"].unique().tolist()

    if roi_width is None:
        try:
            dbfile=get_dbfile(basedir)
            roi_width=load_roi_width(dbfile)
        except AssertionError:
            with open(
                os.path.join(basedir, "metadata.yaml"),
                "r", encoding="utf-8"
            ) as handle:
                roi_width=yaml.load(handle, yaml.SafeLoader)["__store"]["imgshape"][1]


    fig, ax=setup_background()
    fig.canvas.draw()
    clock_size=200
    # Convert canvas data to a numpy array
    bg = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    bg = bg.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    bg = cv2.resize(bg, (clock_size, clock_size))
    # Create a binary mask for non-white pixels
    mask = ~(np.all(bg == [255, 255, 255], axis=-1))

    n_steps=back_in_time//seconds_between
    index=df[["frame_number", "t"]]
    fns = index["frame_number"]
    fns = fns.loc[fns%(framerate*seconds_between)==0].unique()
    fns = fns[fns < chunksize*max_chunk]
    fns = fns[fns >= chunksize*min_chunk]
    fns=fns[n_steps:]
    fns_partition = [
        fns[i*partition_size:(i+1)*partition_size]
        for i in range(fns.size//partition_size+1)
    ]

    out=joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(
            draw_frame
        )(
           df, index, basedir, fns,
           identities=identities,
           step=framerate*seconds_between,
           n_steps=n_steps,
           roi_width=roi_width
        )
        for fns in tqdm(fns_partition)
    )

    imgs_t=[]
    for partition in out:
        for img, t in partition:
            if img is None:
                continue
            imgs_t.append((img, t))
    save_video(imgs_t, output_video, mask=mask, fps=fps, clock_size=None)
