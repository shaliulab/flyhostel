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
    
    # # Drop the 'bout_count' column if you don't need it
    # df.drop(columns=['bout_count'], inplace=True)
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