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


def compute_distance_between_ids(df, identities, **kwargs):
    """
        Arguments:

        * df (cudf.DataFrame): Contains columns id, frame_number, centroid_x, centroid_y.
        The centroid columns must contain the coordinates in units of pixels of the original raw recording

        * identities (list): For each animal in this list, the function will check the distance to all other animals
        in all frames available in df

        Returns
            distance_matrix (cp.array): ids x neighbors x t
        
    """
    distances=[]
    pairs=[]
    min_fn=df["frame_number"].min()
    max_fn=df["frame_number"].max()
    identities=sorted(identities)


    df.reset_index(drop=True).to_feather("df_position.feather")

    for id1, id2 in itertools.combinations(identities, 2):
        with codetiming.Timer(text=f"Done computing distance between {id1} and {id2} in " + "{:.4f} seconds", logger=logger.debug):
            distances.append(
                compute_distance_between_pairs(
                    df,
                    id1, id2,
                    min_fn=min_fn,
                    max_fn=max_fn,
                    **kwargs
                )
            )
            pairs.append((id1, id2))
    
    
    distance_matrix=[]
    for animal1 in identities:
        this_animal_distances=[]
        for animal2 in identities:
            if animal1==animal2:
                continue
            
            this_pair = tuple(sorted([animal1, animal2]))
            selector=pairs.index(this_pair)
            this_animal_distances.append(cp.asnumpy(distances[selector]))
        
        distance_matrix.append(np.stack(this_animal_distances))
        del this_animal_distances
    
    del distances

    distance_matrix=np.stack(distance_matrix)
    
    return distance_matrix



def compute_distance_between_pairs(df, id1, id2, min_fn=None, max_fn=None, useGPU=True, step=1):
    """
    Com
    
    Arguments:
        df (xf.DataFrame): Contains columns id, frame_number, centroid_x, centroid_y.
        id1: id of one of the animals in the pair.
        id2: id of another of the animals in the pair.
        min_fn (int): First frame number where the distance will be computed. By default, all frames available will be used.
        max_fn (int): Last frame number where the distance will be computed. By default, all frames available will be used.
        
    """
    if useGPU:
        xf=cudf
        nx=cp
    else:
        xf=pd
        nx=np
    summands={}
    for coord in ["x", "y"]:
        p0=df.loc[df["id"]==id1, ["frame_number", f"centroid_{coord}"]].set_index("frame_number")
        p1=df.loc[df["id"]==id2, ["frame_number", f"centroid_{coord}"]].set_index("frame_number")
        diff=p1-p0
        diff=diff.loc[~diff[f"centroid_{coord}"].isna()]
        summands[coord]=xf.Series(index=diff.index, data=diff.values.flatten()**2)
    
    summ=None
    for coord, summand in summands.items():
        if summ is None:
            summ=summand
        else:
            summ+=summand
    
    distance=xf.Series(index=summ.index, data=nx.sqrt(summ.values.flatten()))
    # NOTE
    # this happens if the same identity is present more than once in the same frame
    # which occurs if the validation was not perfect
    duplicates=distance.index.duplicated()
    if duplicates.sum()>0:
        logger.error(f"The following frame numbers have duplicated ids: {distance.index[distance.index.duplicated()]}")
        distance=distance.index[~duplicates]

    if min_fn is None:
        min_fn = df["frame_number"].min()
    if max_fn is None:
        max_fn = df["frame_number"].max()

    # full_range_index = cudf.RangeIndex(start=min_index, stop=max_index + 1)
    try:
        distance = distance.reindex(nx.arange(min_fn, max_fn+step, step, dtype=distance.index.dtype), fill_value=nx.inf).values
    except Exception as error:
        raise error
    
    logger.debug("Filled in %s %% of values with inf", 100*nx.isinf(distance).mean())
    return distance


def find_neighbors(dt, dist_max_px, demo=False):
    """
    Annotate neighbors (NN) of each agent at each timestamp

    Arguments
        dt (cudf.DataFrame): Dataset with columns id, frame_number, centroid_x, centroid_y
        dist_max_px (float): Maximum of pixels between two flies considered to be neigbors
    
    Returns
        dt_annotated (cudf.DataFrame): Dataset with same columns as input plus
            nn, distance, distance_mm

            * nn contains the id of the nearest neighbor
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
        dt_cpu=dt.to_pandas()
        useGPU=True

    else:
        nx=np
        xf=pd
        dt_cpu=dt
        ids_cpu=ids
        useGPU=False

    identities=sorted(ids_cpu.unique())

    data_for_computation=dt[["id", "frame_number", "centroid_x", "centroid_y"]]
    del dt
    fn_min=data_for_computation["frame_number"].min()
    fn_max=data_for_computation["frame_number"].max()

    logger.debug("Done")
    

    distance_matrix_ = compute_distance_between_ids(data_for_computation, identities=identities, useGPU=useGPU)
    del data_for_computation

    if nx is np:
        distance_matrix=cp.asnumpy(distance_matrix_)
    else:
        distance_matrix=cp.array(distance_matrix_)

    # ids x neighbors x t
    frame_number=nx.arange(fn_min, fn_max+1)
    assert len(frame_number)==distance_matrix.shape[2]

    neighbor_matrix=distance_matrix<dist_max_px
    
    nns = None
    focal_identities=identities

    for i, this_identity in tqdm(enumerate(focal_identities), desc="Finding nearest neighbors"):
        neighbors=identities.copy()
        neighbors.pop(neighbors.index(this_identity))
        neighbor_idx, frame_pos=nx.where(neighbor_matrix[i,...])


        this_distance=distance_matrix[i, neighbor_idx, frame_pos]
        nearest_neighbors = xf.Series(index=nx.arange(len(neighbors)), data=neighbors).loc[neighbor_idx.astype(int)]

        out = xf.DataFrame({
            "id": this_identity,
            "nn": nearest_neighbors,
            "distance": this_distance,
            "frame_number": frame_number[frame_pos],
        })
        if nns is None:
            nns=out
        else:
            nns=xf.concat([nns, out], axis=0)

    logger.debug("merging")
    if nx is cp:
        dt=xf.DataFrame(dt_cpu)
    else:
        dt=dt_cpu
    dt_annotated = dt.merge(nns, on=["id", "frame_number"])
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

    df1 = neighbors.merge(pose, how="left", left_on=["id","frame_number"], right_on=["id","frame_number"]).sort_values("frame_number")
    df2 = neighbors.drop("id", axis=1).merge(pose, how="left", left_on=["nn","frame_number"], right_on=["id","frame_number"]).drop("nn", axis=1).sort_values("frame_number")
    # assert all(df1["nn"].to_pandas()==df2["id"].to_pandas())
    # assert (df2["frame_number"].values==df1["frame_number"].values).all()


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

    # cell i, j, k contains the distance between bodypart j (of the nn) and k (of the focal id) in the neighbor pair i (ith row of neighbors)

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
