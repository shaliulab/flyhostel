import itertools
import os.path
import pandas as pd

from flyhostel.data.pose.movie_old import connect_bps
from flyhostel.data.bodyparts import bodyparts
from flyhostel.data.pose.sleap import draw_video_row
from flyhostel.data.pose.filters import filter_pose, arr2df
from flyhostel.data.deg import read_label_file
from flyhostel.data.pose.main import FlyHostelLoader
from flyhostel.data.pose.fh_umap import add_n_steps_in_the_past
from flyhostel.data.pose.constants import bodyparts_xy
from flyhostel.data.pose.constants import chunksize as CHUNKSIZE
from flyhostel.data.pose.constants import framerate as FRAMERATE


import cudf
import cupy as cp


BODY_LENGTH=2
ARENA_WIDTH=60
ARENA_HEIGHT=60
target_fps=30
stride=FRAMERATE//target_fps
dsna_folder="/home/vibflysleep/opt/drosophila-social-network-analysis"


# Function to calculate angle in radians
def calculate_angle_gpu(df):
    cudf_df=cudf.DataFrame(df)
    dx = cudf_df['x2'] - cudf_df['x1']
    dy = cudf_df['y2'] - cudf_df['y1']
    angles = cp.asnumpy(cp.arctan2(dy.values, dx.values))
    
    return angles

def calculate_angle_between_parts(pose, part1, part2):
    df=pose[["id", "frame_number", f"{part1}_x", f"{part1}_y", f"{part2}_x", f"{part2}_y"]]
    df.dropna(axis = 0, how = 'any', inplace = True)
    df.columns=["id", "frame_number", "x1", "y1", "x2", "y2"]
    angles=calculate_angle_gpu(df)
    angles_df=df[["id", "frame_number"]]
    angles_df["angle"]=angles
    return angles_df

def fill_index(df, column):
    # Get the union of all frame numbers
    all_frame_numbers = df[column].unique()
    # Function to reindex each group
    def reindex_group(group):
        return group.reindex(all_frame_numbers, fill_value=np.nan, axis=0)
    
    # Group by 'id' and apply the reindexing function
    grouped = df.groupby('id').apply(lambda x: x.set_index(column).reindex(all_frame_numbers)).drop("id", axis=1).reset_index()
    # Sort by 'id' and 'frame_number' for better readability

    grouped = grouped.sort_values(by=['id', column]).reset_index(drop=True)
    return grouped


def write_to_xlsx(df, output_file='output.xlsx'):
    
    # Using ExcelWriter to write multiple sheets
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Group by 'id' and iterate through groups
        for group_name, group_df in df.groupby('id'):
            # Write each group to a different sheet
            df_out=group_df.drop("id", axis=1, errors="ignore")
            print(df_out.shape)
            df_out.to_excel(writer, sheet_name=str(group_name), index=False)
    
    print(f"Excel file '{output_file}' has been created with separate sheets for each 'id'.")



def export_to_dsna(experiment, min_time=7*3600, max_time=17*3600):
    """
    Exports flyhostel centroid and orientation information to drosophila-social-network-analysis format
    https://github.com/milanXpetrovic/drosophila-social-network-analysis
    """
    print(experiment)

    output_folder=f"{dsna_folder}/.data/trackings/flyhostel/{experiment}"
    cache_folder=f"{dsna_folder}/cache"
    cache_file=f"{cache_folder}/{experiment}.feather"
    cache_file2=f"{cache_folder}/{experiment}_indexed.feather"
    os.makedirs(output_folder, exist_ok=True)
    excel_sheet_path=os.path.join(output_folder, f"{experiment}.xlsx")


    if os.path.exists(cache_file):
        dt_angle_full=pd.read_feather(cache_file)
    
    else:
        loader = FlyHostelLoader(experiment, chunks=range(0, 400))
        loader.load_and_process_data(
            min_time=min_time, max_time=max_time,                
            stride=stride,
            cache="/flyhostel_data/cache",
            filters=None,
            useGPU=0
        )

        # Compute fly orientation
        loader.integrate(loader.dt, loader.pose_boxcar, bodyparts)
        dt_with_pose=loader.dt_with_pose[["x", "y","frame_number", "t", "id"] + bodyparts_xy].sort_values(["frame_number", "id"])
        out=dt_with_pose.merge(loader.dt_sleep[["id", "frame_number", "asleep"]], on=["id", "frame_number"])
        dt_angle=dt_with_pose.merge(
            calculate_angle_between_parts(dt_with_pose, "head", "thorax"),
            on=["id", "frame_number"], how="left",
        )[["x", "y", "angle", "id", "frame_number"]]

        # Interpolate orientation
        dt_angle_interp=[]
        for id, group_df in dt_angle.groupby("id"):
            group_df["angle"].interpolate(method="linear", limit_direction="both", inplace=True)
            dt_angle_interp.append(group_df.reset_index(drop=True))
        dt_angle_interp=pd.concat(dt_angle_interp, axis=0).reset_index(drop=True).sort_values(["id", "frame_number"])
        dt_angle_interp.to_feather(cache_file2)


        # generate NaN coordinates when the fly is not observed
        dt_angle_full=fill_index(dt_angle_interp, "frame_number")
        dt_angle_full["major axis len"]=BODY_LENGTH
        dt_angle_full["x"]*=ARENA_WIDTH
        dt_angle_full["y"]*=ARENA_HEIGHT
        dt_angle_full.drop("frame_number", axis=1, inplace=True)

        # format to expected dsna input: pos x pos y and ori
        dt_angle_full.rename({"x": "pos x", "y": "pos y", "angle": "ori"}, axis=1, inplace=True)
        dt_angle_full.to_feather(cache_file)
    

    print(f"Saving ---> {excel_sheet_path}")
    write_to_xlsx(dt_angle_full, excel_sheet_path)


experiments=[
    # "FlyHostel3_6X_2023-09-28_16-00-00",
    "FlyHostel1_6X_2023-09-20_18-00-00",
]


for experiment in experiments:
    export_to_dsna(experiment)

