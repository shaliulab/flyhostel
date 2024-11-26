import argparse
import datetime
import os.path
import logging

import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.metrics import ConfusionMatrixDisplay
import webcolors
import matplotlib.pyplot as plt
from textwrap import wrap

from flyhostel.data.pose.ethogram.utils import find_window_winner, find_window_winner, postprocessing
from flyhostel.data.pose.constants import chunksize, framerate
from flyhostel.data.pose.constants import BEHAVIOR_IDX_MAP


logger = logging.getLogger(__name__)

# behaviors are shown in this order from bottom to top
behaviors=list(BEHAVIOR_IDX_MAP.keys())

# list of colors is taken from https://www.w3.org/TR/SVG11/types.html#ColorKeywords
# in English
RAINBOW_8_COLOR=["448aff","1565c0","009688","8bc34a","ffc107","ff9800","f44336","ad1457"][::-1]
RAINBOW_8_COLOR=[f"#{value}" for value in RAINBOW_8_COLOR]

PALETTE = {
    "walk": RAINBOW_8_COLOR[0],
    "groom": RAINBOW_8_COLOR[1],
    "feed": RAINBOW_8_COLOR[2],
    "background": RAINBOW_8_COLOR[3],
    "inactive+micromovement": RAINBOW_8_COLOR[4],
    "inactive+rejection": RAINBOW_8_COLOR[5],
    "inactive+pe": RAINBOW_8_COLOR[6],
    "inactive": RAINBOW_8_COLOR[7],
}

def get_parser():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to feather dataset")
    ap.add_argument("--output", required=False, type=str, default=".")
    ap.add_argument("--time-window-length", type=int, default=1)
    return ap
    

def main():
    ap=get_parser()
    args=ap.parse_args()
    dataset=pd.read_feather(args.input)
    fig=draw_ethogram(dataset, time_window_length=args.time_window_length, t0=None)
    html_out=os.path.join(args.output, "plot.html")
    json_out=os.path.join(args.output, "plot.json")
    logger.info("Saving to ---> %s", html_out)
    fig.write_html(html_out)
    fig.write_json(json_out)


def bin_behavior_table(df, time_window_length=1, x_var="seconds", t0=None, behavior_col="behavior"):

    df["zt"]=(df["t"]/3600).round(2)
    df["zt_"]=(df["t"]/3600)

    if x_var=="seconds":
        if t0 is None:
            t0=df["frame_number"].min()

        df["t"]=(df["frame_number"]-t0)/framerate
        x_var="t"

    df["chunk"]=df["frame_number"]//chunksize
    df["frame_idx"]=df["frame_number"]%chunksize
    if time_window_length is not None:
        logger.debug("Setting time resolution to %s second(s)", time_window_length)
        df=find_window_winner(df, time_window_length=time_window_length, other_cols=["zt", "zt_", "score", "chunk", "frame_idx"], behavior_col=behavior_col)

    return df, x_var


def bin_behavior_table_v2(df, time_window_length=1, x_var="seconds", t0=None, behavior_col="behavior"):
    """
    Bin a behavioral sequence in order to decrease its temporal resolution while increasing its accuracy
    The binning occurs by selecting the most common behavior of the bin to represent the bin


    Arguments:
        df (pd.DataFrame): Dataset with columns id, frame_number, t, behavior_col and one column for every possible behavior
            * id is required even if the provided data refers to a single fly (just set the same value everywhere)
            * the column for each behavior should contain the score of that behavior 
        time_window_length (int): Number of seconds that should make up each bin
        x_var (str): If equal to seconds, a new t is computed. Set to None if you want to not change t
        t0 (int): t which should be as new zt0. ignored if x_var is not seconds
        behavior_col (str): Column in df which contains the behavioral label 
    """

    df["zt"]=(df["t"]/3600).round(2)
    df["zt_"]=(df["t"]/3600)

    if "score" not in df.columns:
        logger.warning("Score not found. Setting to 1 in all rows of data")
        df["score"]=1

    if x_var=="seconds":
        if t0 is None:
            t0=df["frame_number"].min()

        df["t"]=(df["frame_number"]-t0)/framerate
        x_var="t"

    df["chunk"]=df["frame_number"]//chunksize
    df["frame_idx"]=df["frame_number"]%chunksize
    if time_window_length is not None:
        logger.debug("Setting time resolution to %s second(s)", time_window_length)
        df=find_window_winner(
            df,
            behaviors=df[behavior_col].unique(),
            time_window_length=time_window_length,
            other_cols=["zt", "zt_", "score", "chunk", "frame_idx"],
            behavior_col=behavior_col
        )
        logger.debug("bin_behavior_table_v2 Done")

    bin_columns=["fraction", "fluctuations"]
    return df, x_var, bin_columns


def generate_marker_colors(score, behavior, palette_rgb_, alpha_to_score=True):
    
    # Map transparency of the color to the value of score
    if alpha_to_score:
        alphas = np.interp(score, [0, 1], [0, 1])
    else:
        alphas=[1,]*score.shape[0]

    marker_colors=[]
    for alpha in alphas:
        if behavior in palette_rgb_:
            color=palette_rgb_[behavior]
        else:
            logger.warning("Behavior %s does not have a color. Setting to white", behavior)
            color="rgba(255, 255, 255,"
        
        marker_colors.append(color + str(alpha) + ')')

    return marker_colors


def generate_hover_text(behavior_data, meta_columns):
    text = []
    
    metadata=[behavior_data[c] for c in meta_columns]

    for meta in zip(*metadata):
        row=""
        for i, col in enumerate(meta):
            row+=f"{meta_columns[i]}: {col} "
        text.append(row)

    return text

def add_track_midline(fig, df, x_var, behavior, color="white"):

    fig.add_trace(go.Scatter(
        x=[df[x_var].min(), df[x_var].max()],
        y=[behavior, behavior],
        mode='lines',
        line=dict(color=color, width=1),
        showlegend=False
    ))
    return fig

def add_track_markers(fig, df, x_var, behavior, palette_rgb_, meta_columns, behavior_col="prediction2"):
    behavior_data = df[df[behavior_col] == behavior]
    text=generate_hover_text(behavior_data, meta_columns=meta_columns)
    marker_colors=generate_marker_colors(behavior_data["score"], behavior, palette_rgb_=palette_rgb_)
    trace=go.Scattergl(
            x=behavior_data[x_var],
            y=[behavior] * len(behavior_data),
            mode='markers',
            marker=dict(size=10, color=marker_colors, symbol="square"),
            name=behavior,
            text=text,
            hoverinfo='text',
    )
    fig.add_trace(trace)
    return fig

def palette_to_rgb_(palette, reverse=False, input_type="hex"):

    if reverse:
        # in BGR
        palette_rgb={behavior: getattr(webcolors, f"{input_type}_to_rgb")(v)[:3][::-1] for behavior, v in palette.items()}
    else:
        # in RGB
        palette_rgb={behavior: getattr(webcolors, f"{input_type}_to_rgb")(v) for behavior, v in palette.items()}

    # in RGBA partial string (to be completed witj alpha value )
    palette_rgb_ = {behavior: f'rgba({v[0]}, {v[1]}, {v[2]},' for behavior, v in palette_rgb.items()}
    return palette_rgb_


def draw_ethogram(df, time_window_length=1, x_var="seconds", message=logger.info, t0=None, palette=PALETTE):

    df=df.copy()
    id = df["id"].iloc[0]
    behavior_col="prediction2"
    df, x_var, bin_columns=bin_behavior_table_v2(df, time_window_length=time_window_length, x_var=x_var, t0=t0,behavior_col=behavior_col)
    palette_rgb_=palette_to_rgb_(palette)
    # Get unique behaviors
    found_behaviors = list(set(list(palette.keys()) + df[behavior_col].unique().tolist()))
    for behav in found_behaviors:
        if behav not in behaviors:
            logger.warning("Ignoring behavior %s", behav)

    zt_min=round(df["zt"].min(), 2)
    zt_max=round(df["zt"].max(), 2)
    message("Generating ethogram from %s to %s ZT", zt_min, zt_max)
    meta_columns=[behavior_col, "id",  "frame_number", "chunk","frame_idx", "zt", "score"] + bin_columns

    # Create a figure
    fig = go.Figure()
    # Plot a thin black line for all behaviors throughout the plot
    for behavior in behaviors:
        print(behavior)
        fig=add_track_midline(fig, df, x_var, behavior=behavior)
        fig=add_track_markers(
            fig, df, x_var, behavior=behavior, palette_rgb_=palette_rgb_,
            meta_columns=meta_columns, behavior_col=behavior_col
        )

    date_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # annoying to read 1 seconds in the plot title x)
    if time_window_length==1:
        title=f"Ethogram - {id} - Resolution {time_window_length} second. Generated {date_time}"
    else:
        title=f"Ethogram - {id} - Resolution {time_window_length} seconds. Generated {date_time}"

    fig.update_layout(
        title=title,
        xaxis_title="ZT (hours)",
        yaxis_title="Behavior",
        yaxis=dict(type='category'),
        showlegend=True,
        height=600,
    )

    fig.update_layout(
        template="plotly_dark"
    )
    
    # Set the default range for the x-axis to the first 300 unique values
    default_x_range = [df[x_var].min(), df[x_var].min() + 300]
    center_x = np.mean(default_x_range)
    
    # Configure the x-axis with the default range
    fig.update_layout(
        xaxis=dict(
            range=default_x_range,
            rangeslider=dict(
                visible=True,
                thickness=0.05
            ),
            type="linear"
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

    step=900
    tickvals=np.arange(
        np.floor(df[x_var].min()/step)*step,
        np.ceil(df[x_var].max()/step)*step,
        step
    )
    ticktext=[str(df["zt"].iloc[np.argmin(np.abs(df[x_var]-val))]) for val in tickvals]        
    fig.update_xaxes(
        tickvals=tickvals,
        ticktext=ticktext
        )
    message("Done")
    return fig


def save_confusion_matrix(predictions, y_true, y_pred, output_folder, name="test_confusion_matrix.png", labels=None, title=None):
    disp=ConfusionMatrixDisplay.from_predictions(
        y_true=predictions[y_true],
        y_pred=predictions[y_pred],
        normalize="true",
        xticks_rotation=45,
        labels=labels
    )
    np.savetxt(
        os.path.join(output_folder, "test_confusion_matrix.csv"),
        disp.confusion_matrix, delimiter=",", fmt="%.4e"
    )

    # Create a larger figure to accommodate the long title
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjust the figure size as needed
    # Display the confusion matrix
    disp.plot(ax=ax)
    # Set a long title and wrap it
    plt.title("\n".join(wrap(title, 60)))  # Wrap text after 60 characters
    # Adjust layout
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, name))
