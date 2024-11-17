import itertools
import traceback
import logging
import codetiming

from tqdm.auto import tqdm

import cudf
import cupy as cp
import pandas as pd
import numpy as np

logger=logging.getLogger(__name__)
time_counter=logging.getLogger("time_counter")


def compute_distance_between_all_ids(df, ids, step=10, **kwargs):
    """
        Arguments:

        * df (cudf.DataFrame): Contains columns id, frame_number, centroid_x, centroid_y.
        The centroid columns must contain the coordinates in units of pixels of the original raw recording

        * identities (list): For each animal in this list, the function will check the distance to all other animals
        in all frames available in df

        Returns
            distance_matrix (cp.array): Distance between two flies in a pair, organized by:
            ids: number of flies
            neighbors: number of flies except the focal fly
            t: timestamps
        
    """
    distances=[]
    pairs=[]
    assert len(ids)>1, f"Pass more than 1 id"
    ids=sorted(ids)
    if isinstance(df, cudf.DataFrame):
        nx=cp
    else:
        nx=np

    for i, (id1, id2) in enumerate(itertools.combinations(ids, 2)):
        with codetiming.Timer(
            text=f"Done computing distance between {id1} and {id2} in " + "{:.4f} seconds",
            logger=time_counter.debug
        ):
            dff=df.loc[df["id"].isin([id1, id2])]
            assert dff.shape[0]>0, f"{id1} and {id2} are not present in this dataset"
            dist=compute_distance_between_pairs(
                dff,
                step=step,
                **kwargs
            )
            distances.append(dist)
            pairs.append((id1, id2))


    distance_matrix=[]
    for animal1 in ids:
        this_animal_distances=[]
        for animal2 in ids:
            if animal1==animal2:
                continue

            this_pair = tuple(sorted([animal1, animal2]))
            selector=pairs.index(this_pair)
            this_animal_distances.append(distances[selector])

        distance_matrix.append(nx.stack(this_animal_distances))

        del this_animal_distances

    del distances
    distance_matrix=nx.stack(distance_matrix)
    return distance_matrix


# Function to compute pairwise distances for each frame
def compute_pairwise_distances(df):
    if isinstance(df, cudf.DataFrame):
        xf=cudf
        nx=cp
        ids=df["id"].to_pandas().unique()
    else:
        xf=pd
        nx=np
        ids=df["id"].unique()


    dups=df.duplicated(["frame_number", "id"], keep="first")
    sum_dups=dups.sum()
    if sum_dups>0:
        logger.warning("%s frames contain a duplicated identity", sum_dups)
        df=df.loc[nx.bitwise_not(dups)]

    df = df.pivot(index="frame_number", columns="id", values=["x", "y"])
    df.columns=["id_1_x","id_2_x","id_1_y", "id_2_y"]

    df=df.loc[(
        nx.bitwise_not(df["id_1_x"].isna()) & \
        nx.bitwise_not(df["id_2_x"].isna())
    )]

    distance=nx.sqrt(
        (df["id_1_x"]-df["id_2_x"])**2 +\
        (df["id_1_y"]-df["id_2_y"])**2
    )
    dist_df=xf.DataFrame({
        "distance": distance,
        "id1": ids[0],
        "id2": ids[1],
        "frame_number": df.index
    })

    # Convert result to a dataframe
    return dist_df


def compute_distance_between_pairs(df, step=10):
    """
    Com
    
    Arguments:
        df (xf.DataFrame): Contains columns id, frame_number, centroid_x, centroid_y.
        min_fn (int): First frame number where the distance will be computed. By default, all frames available will be used.
        max_fn (int): Last frame number where the distance will be computed. By default, all frames available will be used.
    
    NOTE: This function supports cudf.DataFrame ops
    """

    if isinstance(df, cudf.DataFrame):
        xf=cudf
        nx=cp
    else:
        xf=pd
        nx=np

    df=df\
        .rename({
            "centroid_x": "x",
            "centroid_y": "y",
        }, axis=1)\

    dist_df=compute_pairwise_distances(
        df[["frame_number", "id", "x", "y"]]
    ).reset_index(drop=True)

    min_fn=df["frame_number"].min()
    max_fn=df["frame_number"].max()

    distance=dist_df.set_index("frame_number")["distance"]
    
    new_index=nx.arange(min_fn, max_fn+step, step)
    old_index=distance.index
    distance=distance.reindex(new_index.astype(old_index.dtype))
    distance.fillna(nx.inf, inplace=True)

    logger.debug("Filled in %s %% of values with inf", 100*nx.isinf(distance).mean())
    return distance.values


def find_neighbors(dt, dist_max_px, step, demo=False):
    """
    Annotate neighbors (NN) of each agent at each timestamp

    Arguments
        dt (cudf.DataFrame): Dataset with columns id, frame_number, centroid_x, centroid_y
        dist_max_px (float): Maximum of pixels between two flies considered to be neigbors
    
    Returns
        dt_annotated (cudf.DataFrame): Dataset with same columns as input
        plus nn, distance:
            * nn contains the id of a near neighbor
            * distance is in pixels
    """

    logger.debug("Downloading identities from GPU")
    ids=dt["id"]
    
    if isinstance(ids, cudf.Series):
        # ids=ids.to_pandas()
        # dt=dt.to_pandas()
        ids_cpu=ids.to_pandas()
        nx=cp
        xf=cudf
    else:
        nx=np
        xf=pd
        ids_cpu=ids

    ids=sorted(ids_cpu.unique().tolist())
    index_single=dt[["frame_number"]].drop_duplicates()
    index=[]
    for id in ids:
        block=index_single.copy()
        block["id"]=id
        index.append(block)
    index=xf.concat(index, axis=0).reset_index(drop=True)
    
    dt=index.merge(
        dt,
        on=["frame_number", "id"],
        how="left"
    ).sort_values("frame_number")

    data_for_computation=dt[["id", "frame_number", "centroid_x", "centroid_y"]]
    del dt
    fn_min=data_for_computation["frame_number"].min()
    fn_max=data_for_computation["frame_number"].max()+step



    distance_matrix=compute_distance_between_all_ids(data_for_computation, ids=ids, step=step)

    # ids x neighbors x t
    frame_number=nx.arange(fn_min, fn_max, step)
    assert len(frame_number)==distance_matrix.shape[2]
    neighbor_matrix=distance_matrix<dist_max_px
    
    nns = []
    focal_ids=ids

    for i, this_identity in tqdm(enumerate(focal_ids), desc="Finding nearest neighbors"):
        other_ids=ids.copy()
        other_ids.pop(other_ids.index(this_identity))

        # this can include 0 1 or more neighbors in the same frame
        neighbor_idx, frame_pos=nx.where(neighbor_matrix[i,...])

        this_distance=distance_matrix[i, neighbor_idx, frame_pos]
        
        nearest_neighbors = xf.Series(
            index=nx.arange(len(other_ids)),
            data=other_ids
        ).loc[neighbor_idx.astype(int)]

        out = xf.DataFrame({
            "id": this_identity,
            "nn": nearest_neighbors,
            "distance": this_distance,
            "frame_number": frame_number[frame_pos],
        })

        nns.append(out)

    nns=xf.concat(nns, axis=0)
    logger.debug("merging")
    dt_annotated = data_for_computation.merge(nns, on=["id", "frame_number"])
    logger.debug("done")
    if demo:
        return dt_annotated, distance_matrix
    else:
        return dt_annotated



def compute_pairwise_distances_using_bodyparts_gpu(neighbors, pose, bodyparts, bodyparts_xy, useGPU=False):
    """
    Compute distance between two closest bodyparts of two already close animals
    """
    if useGPU:
        nx=cp
    
    else:
        nx=np

    df1 = neighbors.merge(
        pose, how="left",
        left_on=["id","frame_number"],
        right_on=["id","frame_number"]
    ).sort_values("frame_number")
    df2 = neighbors.drop("id", axis=1).merge(
        pose, how="left",
        left_on=["nn","frame_number"],
        right_on=["id","frame_number"]
    ).drop("nn", axis=1).sort_values("frame_number")

    bodyparts_1=df1[bodyparts_xy].fillna(nx.inf).values.reshape((-1, len(bodyparts), 2))
    bodyparts_2=df2[bodyparts_xy].fillna(nx.inf).values.reshape((-1, len(bodyparts), 2))


    across_bodyparts=[]
    for bp_i, bp in tqdm(enumerate(bodyparts), desc="Computing distance between bodyparts"):
        diff=bodyparts_1[:,[bp_i],:]-bodyparts_2
        # make inf-inf (which yields nan) also inf
        # that way the distance is inf
        diff[nx.isnan(diff)]=nx.inf

        across_bodyparts.append(
            nx.sqrt(
                ((diff)**2).sum(axis=2)
            )
        )

    across_bodyparts=nx.stack(across_bodyparts, axis=2)
    if across_bodyparts.shape[0]==0:
        return None

    # cell i, j, k contains the distance between bodypart j (of the nn)
    # and k (of the focal id) in the neighbor pair i (ith row of neighbors)

    logger.debug("Finding closest bodyparts")
    distance, (bp_index_2, bp_index_1) = find_closest_entities(across_bodyparts)
    nn_bodypart=cudf.Series(bodyparts)[bp_index_2].reset_index(drop=True)
    id_bodypart=cudf.Series(bodyparts)[bp_index_1].reset_index(drop=True)

    nn_bodypart.index=neighbors.index
    id_bodypart.index=neighbors.index
    

    neighbors["nn_bodypart"]=nn_bodypart
    neighbors["id_bodypart"]=id_bodypart
    neighbors["distance_bodypart"]=distance

    # keep only the rows where a non infinite distance was found
    neighbors=neighbors.loc[~nx.isinf(neighbors["distance_bodypart"])]

    return neighbors.sort_values("frame_number")


def find_closest_entities(arr):
    """
    arr is assumed to have dimensions interactions x bodyparts x bodyparts
    and contains the distance in pixels between two bodyparts in each interaction timepoint
    """
    if isinstance(arr, cp.ndarray):
        nx=cp
    else:
        nx=np
        
        
    # Find the minimum distances for each time slice
    min_distances = nx.min(arr, axis=(1, 2))

    # Reshape the array to 2D (time x (body parts combined))
    reshaped_arr = arr.reshape(arr.shape[0], -1)

    # Find the flattened indices of the minimum values in the reshaped array
    flat_indices = nx.argmin(reshaped_arr, axis=1)

    # Convert flattened indices to 2D indices in the original n x n body parts grid
    n = arr.shape[1]  # Assuming the second and third dimensions are the same
    j_indices, k_indices = nx.divmod(flat_indices, n)

    return min_distances, (j_indices, k_indices)



def find_closest_pair(arr, time_axis, partner_axis):
    """
    arr is assumed to have 3 dimensions
    """
    if isinstance(arr, cp.ndarray):
        nx=cp
    else:
        nx=np

    # Find the minimum distances for each time slice
    min_distances = nx.min(arr, axis=(time_axis, partner_axis))

    focal_axis=[0, 1, 2]
    focal_axis.pop(focal_axis.index(time_axis))
    focal_axis.pop(focal_axis.index(partner_axis))
    focal_axis=focal_axis[0]

    # Reshape the array to 2D (time x (body parts combined))
    reshaped_arr = arr.reshape(arr.shape[focal_axis], -1)

    # Find the flattened indices of the minimum values in the reshaped array
    try:
        flat_indices = nx.argmin(reshaped_arr, axis=1)
    except Exception as error:
        print(error)
        import ipdb; ipdb.set_trace()
    # Convert flattened indices to 2D indices in the original n x n body parts grid
    n=arr.shape[-1] # because of the C order in reshape
    j_indices, k_indices = nx.divmod(flat_indices, n)

    return min_distances, (j_indices, k_indices)


def find_closest_entities_v2(arr):
    return find_closest_pair(arr, time_axis=0, partner_axis=1)
