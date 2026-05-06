from abc import ABC, abstractmethod
import os.path
import hdf5storage
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.io as pio

from flyhostel.utils import get_sqlite_file, get_chunksize

pio.templates.default = "simple_white"
px.defaults.template = "ggplot2"
px.defaults.color_continuous_scale = px.colors.sequential.Blackbody
px.defaults.width = 1000
px.defaults.height = 1000
px.defaults.size_max = 15

MMPY_PROJECT="/home/vibflysleep/opt/long_timescale_analysis/slurm/FlyHostel_long_timescale_analysis/"


class UMAPLoader(ABC):

    def __init__(self, *args, **kwargs):
        self.pose=None
        self.experiment=None
        super(UMAPLoader, self).__init__(*args, **kwargs)

    @abstractmethod
    def load_data(self):
        pass


    def load_umap_data(self, identity, min_time=None, max_time=None):

        animal = self.experiment + "__" + str(identity).zfill(2)
        chunksize=get_chunksize(self.experiment)

        mat_file = f"{MMPY_PROJECT}/Projections/{self.experiment}__{str(identity).zfill(2)}-pcaModes_uVals.mat"
        if not os.path.exists(mat_file):
            print(f"{mat_file} not found. Please run motionmapper pipeline")
            return None
        
        umap_result=hdf5storage.loadmat(mat_file)
        df = pd.DataFrame(umap_result["zValues"], columns=["x", "y"])
    
    
        assert self.pose is not None
        pose=self.pose.copy()
        if min_time is not None:
            pose=pose.loc[pose["t"] >= min_time]
        if max_time is not None:
            pose=pose.loc[pose["t"] < max_time]


        assert df.shape[0] == pose.shape[0], f"{df.shape[0]} != {pose.shape[0]}"
        df["frame_number"]=pose["frame_number"].values

        id = pose["id"].iloc[0]

        if "1X" in id:
            df["id"]=id.split("|")[0] + "|" + str(0).zfill(2)
        else:
            df["id"]=id

        df["chunk"]=df["frame_number"]//chunksize
        df["frame_idx"] = df["frame_number"]%chunksize
        
        df.set_index("frame_number", inplace=True)
        df["frame_number"]=pose["frame_number"].values
        return df


    @staticmethod
    def plot_umap(df, include=None, exclude=None, chunks=None, hover_data=['id', 'chunk', 'frame_idx'], max_rows=45000, downsample=150, show=False):
        
        df_plot = df.copy()
        xlim = [
            np.floor(df_plot["x"].min() / 10)*10,
            np.ceil(df_plot["x"].max() / 10)*10
        ]
        ylim = [
            np.floor(df_plot["y"].min() / 10)*10,
            np.ceil(df_plot["y"].max() / 10)*10
        ]
        
        if include is not None:
            if isinstance(include, str):
                include=[include]
            df_plot=df_plot.loc[df_plot["behavior"].isin(include)]
        
        if exclude is not None:
            if isinstance(exclude, str):
                exclude=[exclude]
            df_plot=df_plot.loc[~df_plot["behavior"].isin(exclude)]
            
        if chunks is not None:
            df_plot=df_plot.loc[df_plot["chunk"].isin(chunks)]
            
        
        if df_plot.shape[0]>max_rows:
            df_plot=df_plot.iloc[::downsample]
        
        df_plot["size"]=5
        df_plot["size"].loc[df_plot["behavior"]=="unknown"]=1
        fig = px.scatter(
            df_plot, x='x', y='y', color="behavior", hover_data=hover_data,
            size="size", color_discrete_sequence=px.colors.qualitative.Plotly

        )
        fig.update_traces(marker={
            "line_width": 0
        })
        # Set x-axis range
        fig.update_xaxes(range=xlim, title="UMAP 1")  # Replace xmin and xmax with your actual values

        # Set y-axis range
        fig.update_yaxes(range=ylim, title="UMAP 2")  # Replace ymin and ymax with your actual values
        
        if show:
            # Show the plot
            fig.show()
        return fig