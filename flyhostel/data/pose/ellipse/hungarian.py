import logging
import numpy as np
from scipy.optimize import linear_sum_assignment

logger=logging.getLogger(__name__)

def hungarian_matching_sf(df_ellipses, df_ids, frame):
    # Extract data for the current frame
    ellipses_frame = df_ellipses[df_ellipses['frame_number'] == frame].copy()
    ids_frame = df_ids[df_ids['frame_number'] == frame].copy()
    
    # Skip if there are no ellipses or IDs in this frame
    if ellipses_frame.empty or ids_frame.empty:
        return
    
    # Get coordinates
    ellipses_coords = ellipses_frame[['x', 'y']].values
    ids_coords = ids_frame[['x', 'y']].values
    
    # Compute the distance matrix between ellipses and IDs
    distances = np.linalg.norm(ellipses_coords[:, np.newaxis] - ids_coords[np.newaxis, :], axis=2)
    
    # Set distances greater than 2 units to a large number to avoid matching
    distance_threshold = 2  # You can adjust this threshold as needed
    distances[distances > distance_threshold] = np.inf

    try:
        # Use the Hungarian Algorithm to find the optimal assignment
        row_ind, col_ind = linear_sum_assignment(distances)
    except ValueError as error:
        # TODO Adjust code so it actually checks the content of the message
        if True or error.endswith("cost matrix is infeasible"):
            logger.debug(frame, error)
            row_ind=[]
            col_ind=[]
            for ellipse_idx in range(distances.shape[1]):
                idx=np.argmin(distances[:, ellipse_idx])
                row_ind.append(idx)
                col_ind.append(ellipse_idx)
        else:
            raise error

    # Assign IDs to ellipses based on the optimal assignment
    for r, c in zip(row_ind, col_ind):
        if distances[r, c] == np.inf:
            # Distance exceeds threshold; do not assign this ID
            continue
        # Get the index of the ellipse in the original DataFrame
        ellipse_index = ellipses_frame.index[r]
        # Get the corresponding ID
        id_value = ids_frame.iloc[c]['id']
        # Assign the ID to the ellipse
        df_ellipses.loc[ellipse_index, 'id'] = id_value
    return df_ellipses

    
def hungarian_matching(df_ellipses, df_ids, nan_policy="propagate"):
    # Assume df_ellipses is your first dataset with columns: x, y, major, minor, angle, frame_number
    # Assume df_ids is your second dataset with columns: id, frame_number, x, y

    # Add an 'id' column to df_ellipses to store the assigned IDs
    df_ellipses=df_ellipses.copy()
    df_ellipses['id'] = np.nan
    
    # Get unique frame numbers
    frame_numbers = df_ellipses['frame_number'].unique()

    df_ids=df_ids.rename({
        "center_x": "x",
        "center_y": "y"
    }, axis=1).copy()

    
    # Loop over each frame
    for frame in frame_numbers:
        df_ellipses=hungarian_matching_sf(df_ellipses, df_ids, frame)
    
    missing=df_ellipses["id"].isna()

    if missing.mean()>0:
        if nan_policy=="omit":
            df_ellipses=df_ellipses.loc[~missing]
        elif nan_policy=="raise":
            raise Exception("Not all ellipses have a matching id")
    return df_ellipses
