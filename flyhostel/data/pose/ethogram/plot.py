import argparse
import datetime
import os.path
import logging

import pandas as pd
import numpy as np
import plotly.graph_objs as go
import webcolors

from flyhostel.data.pose.ethogram.utils import find_window_winner, find_window_winner, postprocessing
from flyhostel.data.pose.constants import chunksize, framerate


logger = logging.getLogger(__name__)

# behaviors are shown in this order from bottom to top
behaviors=[
    "inactive",
    "inactive+microm",
    "inactive+micromovement",
    "inactive+twitch",
    "pe_inactive",
    "feed+inactive",
    "feed",
    "feed+groom",
    "groom+pe",
    "groom",
    "feed+walk",
    "walk",
    "background",
]

# list of colors is taken from https://www.w3.org/TR/SVG11/types.html#ColorKeywords
# in English
PALETTE = {
    "pe_inactive": "gold",
    "feed": "orange",
    "groom": "green",
    "inactive": "blue",
    "walk": "red",
    "background": "white",
    "feed+inactive": "peachpuff",
    "groom+pe": "darkseagreen",
    "feed+walk": "crimson",
    "feed+groom": "peachpuff",
    "inactive+walk": "darkgray",
    "inactive+microm": "purple",
    "inactive+micromovement": "purple",
    "inactive+twitch": "purple",    
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
        df=find_window_winner(
            df,
            behaviors=df[behavior_col].unique(),
            time_window_length=time_window_length,
            other_cols=["zt", "zt_", "score", "chunk", "frame_idx"],
            behavior_col=behavior_col
        )

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

def add_track_black_line(fig, df, x_var, behavior):

    fig.add_trace(go.Scatter(
        x=[df[x_var].min(), df[x_var].max()],
        y=[behavior, behavior],
        mode='lines',
        line=dict(color='black', width=1),
        showlegend=False
    ))
    return fig

def add_track_markers(fig, df, x_var, behavior, palette_rgb_, meta_columns):
    behavior_data = df[df['behavior'] == behavior]
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

def palette_to_rgb_(palette, reverse=False):

    if reverse:
        # in BGR
        palette_rgb={behavior: webcolors.name_to_rgb(v)[:3][::-1] for behavior, v in palette.items()}
    else:
        # in RGB
        palette_rgb={behavior: webcolors.name_to_rgb(v) for behavior, v in palette.items()}

    # in RGBA partial string (to be completed witj alpha value )
    palette_rgb_ = {behavior: f'rgba({v[0]}, {v[1]}, {v[2]},' for behavior, v in palette_rgb.items()}
    return palette_rgb_


def draw_ethogram(df, time_window_length=1, x_var="seconds", message=logger.info, t0=None, palette=PALETTE):

    df=df.copy()
    id = df["id"].iloc[0]
    df, x_var, bin_columns=bin_behavior_table_v2(df, time_window_length=time_window_length, x_var=x_var, t0=t0)
    

    df=postprocessing(df, time_window_length=time_window_length)
    df["behavior_30fps"]=df["behavior"].copy()

    
    palette_rgb_=palette_to_rgb_(palette)
    # Get unique behaviors
    found_behaviors = list(set(list(palette.keys()) + df["behavior"].unique().tolist()))
    for behav in found_behaviors:
        if behav not in behaviors:
            logger.warning("Ignoring behavior %s", behav)

    zt_min=round(df["zt"].min(), 2)
    zt_max=round(df["zt"].max(), 2)
    message("Generating ethogram from %s to %s ZT", zt_min, zt_max)
    meta_columns=["behavior", "id", "frame_number", "chunk","frame_idx", "zt", "score"] + bin_columns

    # Create a figure
    fig = go.Figure()
    # Plot a thin black line for all behaviors throughout the plot
    for behavior in behaviors:
        fig=add_track_black_line(fig, df, x_var, behavior=behavior)
        fig=add_track_markers(fig, df, x_var, behavior=behavior, palette_rgb_=palette_rgb_, meta_columns=meta_columns)

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