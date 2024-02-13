import numpy as np
from flyhostel.data.pose.constants import framerate as FRAMERATE

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

    return df


def annotate_bout_duration(dataset, fps=FRAMERATE):
    duration_table=dataset.loc[dataset["bout_out"]==1, ["bout_in", "bout_count"]]
    duration_table["duration"]=duration_table["bout_in"]/fps
    dataset=dataset.drop("duration", axis=1, errors="ignore").merge(duration_table.drop("bout_in", axis=1), on=["bout_count"])
    return dataset

def annotate_bouts(dataset, variable):
    dataset=count_bout_position(dataset.iloc[::-1], variable=variable, counter="bout_out").iloc[::-1]
    dataset=count_bout_position(dataset, variable=variable, counter="bout_in")
    return dataset


def most_common_behavior_vectorized(df, time_window_length=1, other_cols=[]):
    # Adjust time to the nearest time_window_length
    df['t'] = (df['t'] // time_window_length) * time_window_length

    index=df[["id", "t", "frame_number"] + other_cols].groupby(["id", "t"]).first().reset_index()
    
    # Group by 'id' and 't', and get the most common behavior
    most_common_behavior = df.groupby(['id', 't', 'behavior']).size()
    most_common_behavior = most_common_behavior.reset_index(name='count')
    most_common_behavior = most_common_behavior.sort_values(['id', 't', 'count'], ascending=[True, True, False])
    
    # Drop duplicates to keep the most common behavior for each group
    most_common_behavior = most_common_behavior.drop_duplicates(subset=['id', 't'])
    
    # Calculate the fraction of the most common behavior
    total_counts = df.groupby(['id', 't']).size().reset_index(name='total_count')
    most_common_behavior = most_common_behavior.merge(total_counts, on=['id', 't'])
    most_common_behavior['fraction'] = np.round(most_common_behavior['count'] / most_common_behavior['total_count'], 2)

    # Select the required columns
    most_common_behavior = most_common_behavior[['id', 't', 'behavior', 'fraction']]

    most_common_behavior=most_common_behavior.merge(index, on=["id", "t"])
    return most_common_behavior
