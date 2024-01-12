import logging
import glob
import math
import logging
import os.path
import pickle
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from flyhostel.data.pose.constants import chunksize, framerate

logger=logging.getLogger(__name__)
try:
    from deepethogram.postprocessing import remove_short_bouts_from_trace, get_bout_length_percentile, compute_background
    from deepethogram import projects, file_io
except ModuleNotFoundError:
    remove_short_bouts_from_trace=None
    projects=None
    file_io=None
    logger.error("Please install deepethogram without dependencies (pip install --no-deps deepethogram")

logger = logging.getLogger(__name__)


MOTIONMAPPER_DATA=os.environ["MOTIONMAPPER_DATA"]
DEEPETHOGRAM_PROJECT_PATH=os.environ["DEEPETHOGRAM_PROJECT_PATH"]

FLYHOSTEL_VIDEOS=os.environ["FLYHOSTEL_VIDEOS"]
OUTPUT_FOLDER=os.path.join(MOTIONMAPPER_DATA, "output")
MODELS_PATH=os.path.join(MOTIONMAPPER_DATA, "models")
stride=25
MODEL_PATH=glob.glob(os.path.join(MODELS_PATH, "*knn.pkl"))[0]


def get_bout_length_percentile_from_project(project_path, percentile=1):
    records = projects.get_records_from_datadir(os.path.join(project_path, "DATA"))
    label_list = []
    behaviors=projects.get_classes_from_project(project_path)
    for animal, record in records.items():
        labelfile = record['label']
        if labelfile is None:
            continue
        label = file_io.read_labels(labelfile)
        # ignore partially labeled videos
        if np.any(label == -1):
            continue
        label_list.append(label)
    
    percentiles = get_bout_length_percentile(label_list, percentile)
    percentiles={behaviors[i]: percentiles[i] for i in range(len(percentiles))}
    return percentiles



def count_bout_position(df, variable, counter):
    """
    Compute position in bout of variable

    Detect bouts of rows wit the same value in column variable
    and annotate the position of each row in its bout
    """

    # Initialize a new column with zeros
    df[counter] = 0

    # Create a mask to identify the start of each bout
    bout_start_mask = df[variable] != df[variable].shift(1)
    
    # Create a cumulative count of bout starts
    df['bout_count'] = bout_start_mask.cumsum()
    
    # Calculate the rank within each bout
    df[counter] = df.groupby('bout_count').cumcount() + 1
    
    # # Drop the 'bout_count' column if you don't need it
    # df.drop(columns=['bout_count'], inplace=True)
    return df


def annotate_bout_duration(dataset, fps=framerate/stride):
    duration_table=dataset.loc[dataset["bout_out"]==1, ["bout_in", "bout_count"]]
    duration_table["duration"]=duration_table["bout_in"]/fps
    dataset=dataset.drop("duration", axis=1, errors="ignore").merge(duration_table.drop("bout_in", axis=1), on=["bout_count"])
    return dataset

def annotate_bouts(dataset, variable):
    dataset=count_bout_position(dataset.iloc[::-1], variable=variable, counter="bout_out").iloc[::-1]
    dataset=count_bout_position(dataset, variable=variable, counter="bout_in")
    return dataset

def generate_path_to_output_folder(experiment, identity):
    tokens=experiment.split("_")
    key=os.path.sep.join([tokens[0], tokens[1], "_".join(tokens[2:4])])
    basedir=os.path.join(FLYHOSTEL_VIDEOS, key)
    output_folder=os.path.join(basedir, "motionmapper", str(identity).zfill(2))
    return output_folder

def generate_path_to_data(experiment, identity):
    folder=generate_path_to_output_folder(experiment, identity)
    data_src=os.path.join(folder, experiment + "__" + str(identity).zfill(2) + ".csv")
    assert os.path.exists(data_src), f"{data_src} not found"
    return data_src


# Group by 't' and find the most common 'foo' for each group
# Modified function to include data from all other columns in the first row of each group
def most_common(group, variable="behavior"):
    most_common_val = group[variable].value_counts().idxmax()
    score = group[variable].value_counts().max() / len(group)
    
    # Select the first row of the group for additional data
    first_row = group.iloc[0]
    
    # Prepare the result including additional data
    result = first_row.to_dict()
    result[variable] = most_common_val
    result['score'] = np.round(score, 2)
    
    return pd.Series(result)


def draw_ethogram(df, resolution=1):
    """
    
    """

    df=df.copy()

    id = df["id"].iloc[0]
    if resolution is not None:
        df["t"]=np.floor(df["t"]//resolution)*resolution
        df = df.groupby(["id", "t"]).apply(most_common).reset_index(drop=True)

    df["zt"]=(df["t"]/3600).round(2)
    df["zt_"]=(df["t"]/3600)

    # Get unique behaviors
    behaviors = df['behavior'].unique()

    x_var="zt_"
    # Create a figure
    fig = go.Figure()

    # Plot a thin black line for all behaviors throughout the plot
    for behavior in behaviors:
        fig.add_trace(go.Scatter(
            x=[df[x_var].min(), df[x_var].max()],
            y=[behavior, behavior],
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False
        ))

    df["zt"]=np.round(df["t"]/3600, 2).values

    # Define a color map for different behaviors
    colors = {behavior: f'rgba({np.random.randint(0, 255)}, {np.random.randint(0, 255)}, {np.random.randint(0, 255)},' for behavior in behaviors}

    # Plot each behavior on a separate track with additional information on hover
    for behavior in behaviors:
        behavior_data = df[df['behavior'] == behavior]

        text = []
        meta_columns=["behavior", "id", "chunk","frame_idx", "zt", "score"]
        metadata=[behavior_data[c] for c in meta_columns]

        for meta in zip(*metadata):
            row=""
            for i, col in enumerate(meta):
                row+=f"{meta_columns[i]}: {col} "
            text.append(row)

        # Map transparency of the color to the value of score
        alphas = np.interp(behavior_data['score'], [0, 1], [0, 1])
        marker_colors = [colors[behavior] + str(alpha) + ')' for alpha in alphas]

        fig.add_trace(go.Scatter(
            x=behavior_data[x_var],
            y=[behavior] * len(behavior_data),
            mode='markers',
            marker=dict(size=10, color=marker_colors, symbol="square"),
            name=behavior,
            text=text,
            hoverinfo='text',
        ))

    # annoying to read 1 seconds in the plot title x)
    if resolution==1:
        title=f"Ethogram - {id} - Resolution {resolution} second"
    else:
        title=f"Ethogram - {id} - Resolution {resolution} seconds"


    fig.update_layout(
        title=title,
        xaxis_title="ZT (hours)",
        yaxis_title="Behavior",
        yaxis=dict(type='category'),
        showlegend=True,
        height=300,

    )

    return fig


def enforce_behavioral_context(dataset, modify, context, replacement, seconds=5, framerate=1):
    """
    Makes sure the behavior surrounding another behavior for n seconds is fixed to the behaviors provided in context

    This is useful to for example, correct spurious bouts of pe_inactive which are not surrounded by pe_inactive
    (although the animal might correctly be presumed to be inactive _during_ the bout)

    dataset=enforce_behavioral_context(dataset, modify="pe_inactive", context=["inactive"], replacement="pe", seconds=5, framerate=1)
    """
    
    n = int(seconds * framerate) # Number of rows to consider before first and after last 'foo'
    
    # Find all rows where behavior is set to 'foo'
    modify_rows = dataset['behavior'] == modify

    behavior_bouts=dataset.loc[modify_rows]
    bouts=behavior_bouts["bout_count"].unique()
    

    # bout_start=df.loc[(modify_rows) & (df["bout_in"]==1), "frame_number"].values
    # bout_end=df.loc[(modify_rows) & (df["bout_out"]==1), "frame_number"].values
    bout_start=np.where(np.bitwise_and(modify_rows, dataset["bout_in"]==1))[0].tolist()
    bout_end=np.where(np.bitwise_and(modify_rows, dataset["bout_out"]==1))[0].tolist()
    assert len(bout_start)==len(bout_end) == len(bouts)

    suppressed=[]
    for i, (start_idx, end_idx) in enumerate(tqdm(zip(bout_start, bout_end), desc=f"Verifying {modify} bouts", total=len(bouts))):
        before_test=(dataset.iloc[(start_idx-n):start_idx]["behavior"].isin(context)).all()
        after_test=(dataset.iloc[end_idx:(end_idx+n)]["behavior"].isin(context)).all()
        if before_test and after_test:
            suppressed.append(False)
        else:
            suppressed.append(True)
            bout_count=bouts[i]
            dataset.loc[(dataset["behavior"]==modify) & (dataset["bout_count"]==bout_count), "behavior"] = replacement

    return dataset

    

def draw_umap(dataset, max_points=50_000):
    
    if dataset.shape[0] > max_points:
        skip = math.ceil(dataset.shape[0] / max_points)
        dataset=dataset.iloc[::skip]

    fig=px.scatter(
        dataset, x="C_1", y="C_2", color="behavior",
        hover_data=["id", "chunk", "frame_idx", "zt", "behavior"],
    )

    return fig


def compute_bout_length_percentile(project_path, percentile=1):
    records = projects.get_records_from_datadir(os.path.join(project_path, "DATA"))
    label_list = []
    behaviors=projects.get_classes_from_project(project_path)
    for animal, record in records.items():
        labelfile = record['label']
        if labelfile is None:
            continue
        label = file_io.read_labels(labelfile)
        # ignore partially labeled videos
        if np.any(label == -1):
            continue
        label_list.append(label)
    
    percentiles = get_bout_length_percentile(label_list, percentile)
    percentiles={behaviors[i]: percentiles[i] for i in range(len(percentiles))}
    return percentiles


def one_hot_encoding(strings, unique_strings):
    # Map each unique string to its index
    index_map = {string: index for index, string in enumerate(unique_strings)}

    # Initialize the encoding table with zeros
    encoding_table = [[0] * len(unique_strings) for _ in strings]

    # Set the corresponding column to 1
    for i, string in enumerate(strings):
        if string in index_map:  # Only if the string is in the unique strings
            encoding_table[i][index_map[string]] = 1

    return np.array(encoding_table)


def join_strings_by_repeated_integers(integers, strings):
    """
    join_strings_by_repeated_integers(rows, prediction)
    """
    grouped_strings = {}
    for integer, string in zip(integers, strings):
        if integer in grouped_strings:
            grouped_strings[integer] += "+" + string
        else:
            grouped_strings[integer] = string
    return list(grouped_strings.values())


def make_ethogram(experiment, identity, model_path, input=None, output="./", frame_numbers=None, postprocess=True):
    """
    Generate ethogram for a particular fly

    experiment
    identity
    model_path: Path to pkl file which when loaded yields
     1) an object with a predict method
     2) a list of columns that should be used as features
    """

    with open(model_path, "rb") as handle:
        classifier, knn_feats=pickle.load(handle)
        
    if input is None:
        if output is None:
            output_folder=generate_path_to_output_folder(experiment, identity)
        else:
            output_folder=output
        csv_path=generate_path_to_data(experiment, identity)
    else:
        output_folder=output
        csv_path=input
    
    assert os.path.exists(csv_path), f"{csv_path} not found. Did you run UMAP model?"
    dataset=pd.read_csv(csv_path, index_col=0)
    logger.debug("Read dataset of shape %s", dataset.shape)
    dataset["chunk"]=dataset["frame_number"]//chunksize
    dataset["frame_idx"]=dataset["frame_number"]%chunksize
    dataset["zt"]=(dataset["t"]/3600).round(2)
    
    if frame_numbers is not None:
        dataset=dataset.loc[
            dataset["frame_number"].isin(frame_numbers)
        ]
    logger.debug("Predicting behavior of %s rows", dataset.shape[0])
    
    dataset["behavior"]=classifier.predict(dataset[knn_feats].values)

    if postprocess:
        unique_behaviors=dataset["behavior"].unique().tolist()
        if "background" in unique_behaviors:
            unique_behaviors.pop(unique_behaviors.index("background"))
        unique_behaviors=["background"] + unique_behaviors
        predictions=one_hot_encoding(dataset["behavior"], unique_behaviors)

        bout_length_dict=get_bout_length_percentile_from_project(DEEPETHOGRAM_PROJECT_PATH, percentile=1)
        bout_length_dict["pe_inactive"]=1
        bout_length_dict["inactive"]=6
        logger.debug("Bout length cutoff %s", bout_length_dict)
        
        bout_lengths=[int(bout_length_dict.get(behav, 1)) for behav in unique_behaviors]
        predictions_smoothed = []
        T, K = predictions.shape
        for i in range(K):
            trace = predictions[:, i]
            trace = remove_short_bouts_from_trace(trace, bout_lengths[i])
            predictions_smoothed.append(trace)
        predictions = np.stack(predictions_smoothed, axis=1)

        predictions = compute_background(predictions)

        rows,cols=np.where(predictions==1)
        prediction=[unique_behaviors[i] for i in cols]
        prediction=join_strings_by_repeated_integers(rows, prediction)
        dataset["behavior"]=prediction


    dataset=annotate_bouts(dataset, "behavior")
    dataset=annotate_bout_duration(dataset)
    # dataset=enforce_behavioral_context(dataset, modify="pe_inactive", context=["inactive", "background", "pe_inactive"], replacement="pe_unknown", framerate=1, seconds=3)
    # dataset=annotate_bouts(dataset, "behavior")
    # dataset=annotate_bout_duration(dataset)

    csv_out=os.path.join(output_folder, "dataset_out.csv")
    logger.info("Saving to ---> %s", csv_out)
    dataset.to_csv(csv_out)

    # bouts_df=dataset.loc[dataset["bout_in"]==1][["id", "chunk", "frame_idx", "t", "C_1","C_2", "behavior", "duration"]]

    fig = draw_umap(dataset, max_points=50_000)
    
    if experiment is not None:
        fig.write_html(os.path.join(output_folder, f"{experiment}__{str(identity).zfill(2)}_umap.html"))
        chunks = sorted(dataset["chunk"].unique())
        for chunk in chunks:
            fig = draw_ethogram(dataset.loc[dataset["chunk"]== chunk], resolution=1)
            fig.write_html(os.path.join(output_folder, f"{experiment}__{str(identity).zfill(2)}_{str(chunk).zfill(6)}_ethogram.html"))
    else:
        fig.write_html(os.path.join(output_folder, f"plot_out.html"))

    return fig