import itertools

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def interaction_dtw(df, target_length=5, angular_columns=[]):
    """
    Compress an interaction into a fixed number of rows.
    
    Parameters:
        df (pd.DataFrame): The original interaction data.
        target_length (int): The number of rows to compress to.
        angular_columns (list): List of column names representing angular data (-180° to 180°).
    
    Returns:
        pd.DataFrame: The compressed interaction.
    """
    # Define the target time points for resampling
    target_time = np.linspace(df['t'].min(), df['t'].max(), target_length)
    
    compressed = {}
    
    for col in df.columns:
        if col == 't':  # Keep time points as is
            compressed[col] = target_time
        elif col in angular_columns:  # Handle angular columns
            # Convert to Cartesian space
            x = np.cos(np.radians(df[col]))
            y = np.sin(np.radians(df[col]))
            
            # Interpolate x and y
            interp_x = interp1d(df['t'], x, kind='linear')(target_time)
            interp_y = interp1d(df['t'], y, kind='linear')(target_time)
            
            # Convert back to angles
            compressed[col] = np.degrees(np.arctan2(interp_y, interp_x))
        else:  # Handle regular numerical columns
            compressed[col] = interp1d(df['t'], df[col], kind='linear')(target_time)
    
    return pd.DataFrame(compressed)

def compress_interaction(df, features, target_length=5):
    compressed=interaction_dtw(df[features], target_length=target_length, angular_columns=["inter_orientation"])
    compressed.drop("t", axis=1, inplace=True)
    columns=list(itertools.chain(*[[f"{column}_{i}" for column in compressed.columns] for i in range(target_length)]))
    compressed=pd.DataFrame([compressed.values.flatten()], columns=columns)
    return compressed