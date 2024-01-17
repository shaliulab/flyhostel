import itertools
import logging
import codetiming

from tqdm.auto import tqdm

import cudf
import cupy as cp

logger=logging.getLogger(__name__)

def compute_distance_between_ids(pose, identities):
    """
    Arguments:

    * pose (cudf.DataFrame): Contains columns id, frame_number, centroid_x and centroid_y.
    The centroid columns must contain the coordinates in units of pixels of the original raw recording

    * identities (list): For each animal in this list, the function will check the distance to all other animals
    in all frames available in pose
    """
    distances=[]
    pairs=[]
    min_fn=pose["frame_number"].min()
    max_fn=pose["frame_number"].max()

    
    for id1, id2 in itertools.combinations(identities, 2):
        with codetiming.Timer(text=f"Done computing distance between {id1} and {id2} in " + "{:.4f} seconds"):
            distances.append(
                compute_distance_between_pairs(
                    pose,
                    id1, id2,
                    min_fn=min_fn,
                    max_fn=max_fn,
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
            this_animal_distances.append(distances[selector])
        
        distance_matrix.append(cp.stack(this_animal_distances))
    distance_matrix=cp.stack(distance_matrix)
    return distance_matrix


def compute_distance_between_pairs(df, id1, id2, min_fn=None, max_fn=None):
    summands={}
    for coord in ["x", "y"]:
        p0=df.loc[df["id"]==id1, ["frame_number", f"centroid_{coord}"]].set_index("frame_number")
        p1=df.loc[df["id"]==id2, ["frame_number", f"centroid_{coord}"]].set_index("frame_number")
        diff=p1-p0
        diff=diff.loc[~diff[f"centroid_{coord}"].isna()]
        summands[coord]=cudf.Series(index=diff.index, data=diff.values.flatten()**2)
    
    summ=None
    for coord, summand in summands.items():
        if summ is None:
            summ=summand
        else:
            summ+=summand
    
    distance=cudf.Series(index=summ.index, data=cp.sqrt(summ.values.flatten()))
    

    # Step 1 & 2: Creating a range of indices from min to max of original index
    if min_fn is None:
        min_fn = distance.index.min()
    if max_fn is None:
        max_fn = distance.index.max()
    
    # full_range_index = cudf.RangeIndex(start=min_index, stop=max_index + 1)
    distance = distance.reindex(cp.arange(min_fn, max_fn+1, 1, dtype=cp.int32), fill_value=cp.inf).values
    return distance



def find_neighbors(dt, dist_max_px):
    """
    Annotate neighbors (NN) of each agent at each timestamp

    Arguments
        dt (cudf.DataFrame): Dataset with columns id, frame_number, centroid_x, centroid_y
    
    Returns
        dt_annotated (cudf.DataFrame): Dataset with same columns as input plus
            nn, distance, distance_mm

            * nn contains the id of the nearest neighbor
            * distance is in pixels
    """

    logger.debug("Downloading identities from GPU")
    identities=sorted(dt["id"].to_pandas().unique())
    logger.debug("Done")
    distance_matrix = compute_distance_between_ids(dt, identities=identities)
    # ids x neighbors x t
    frame_number=cp.arange(dt["frame_number"].min(), dt["frame_number"].max()+1)
    assert len(frame_number)==distance_matrix.shape[2]

    neighbor_matrix=distance_matrix<dist_max_px
    
    nns = None
    focal_identities=identities

    for i, this_identity in tqdm(enumerate(focal_identities), desc="Finding nearest neighbors"):
        neighbors=identities.copy()
        neighbors.pop(neighbors.index(this_identity))
        neighbor_idx, frame_pos=cp.where(neighbor_matrix[i,...])


        this_distance=distance_matrix[i, neighbor_idx, frame_pos]
        nearest_neighbors = cudf.Series(index=cp.arange(len(neighbors)), data=neighbors).loc[neighbor_idx.astype(int)]

        out = cudf.DataFrame({
            "id": this_identity,
            "nn": nearest_neighbors,
            "distance": this_distance,
            "frame_number": frame_number[frame_pos],
        })
        if nns is None:
            nns=out
        else:
            nns=cudf.concat([nns, out], axis=0)

    logger.debug("merging")
    dt_annotated = dt.merge(nns, on=["id", "frame_number"])
    logger.debug("done")
    return dt_annotated



def compute_pairwise_distances_using_bodyparts_gpu(neighbors, pose, bodyparts, bodyparts_xy):
    """
    Compute distance between two closest bodyparts of two already close animals
    """

    df1 = neighbors.merge(pose, how="left", left_on=["id","frame_number"], right_on=["id","frame_number"]).sort_values("frame_number")
    df2 = neighbors.drop("id", axis=1).merge(pose, how="left", left_on=["nn","frame_number"], right_on=["id","frame_number"]).drop("nn", axis=1).sort_values("frame_number")
    # assert all(df1["nn"].to_pandas()==df2["id"].to_pandas())
    # assert (df2["frame_number"].values==df1["frame_number"].values).all()


    bodyparts_1=df1[bodyparts_xy].fillna(cp.inf).values.reshape((-1, len(bodyparts), 2))
    bodyparts_2=df2[bodyparts_xy].fillna(cp.inf).values.reshape((-1, len(bodyparts), 2))
    
    # bodyparts_1=df1[bodyparts_xy].fillna(cp.inf).values
    # bodyparts_1=cp.stack([bodyparts_1[:,::2], bodyparts_1[:,1::2]], axis=2)

    # bodyparts_2=df2[bodyparts_xy].fillna(cp.inf).values
    # bodyparts_2=cp.stack([bodyparts_2[:,::2], bodyparts_2[:,1::2]], axis=2)
    

    across_bodyparts=[]
    for bp_i, bp in tqdm(enumerate(bodyparts), desc="Computing distance between bodyparts"):
        diff=bodyparts_1[:,[bp_i],:]-bodyparts_2
        across_bodyparts.append(
            cp.sqrt(
                ((diff)**2).sum(axis=2)
            )
        )
    
    across_bodyparts=cp.stack(across_bodyparts, axis=2)

    # cell i, j, k contains the distance between bodypart j (of the nn) and k (of the focal id) in the neighbor pair i (ith row of neighbors)

    logger.debug("Finding closest bodyparts")
    distance, (bp_index_2, bp_index_1) = find_closest_entities(across_bodyparts)

    # print(across_bodyparts[idx, 6 ,7])

    nn_bodypart=cudf.Series(bodyparts)[bp_index_2].reset_index(drop=True)
    id_bodypart=cudf.Series(bodyparts)[bp_index_1].reset_index(drop=True)

    nn_bodypart.index=neighbors.index
    id_bodypart.index=neighbors.index
    

    neighbors["nn_bodypart"]=nn_bodypart
    neighbors["id_bodypart"]=id_bodypart
    neighbors["distance_bodypart"]=distance

    return neighbors.sort_values("frame_number")


def find_closest_entities(arr):
    """
    arr is assumed to have dimensions interactions x bodyparts x bodyparts
    """
    # Find the minimum distances for each time slice
    min_distances = cp.nanmin(arr, axis=(1, 2))

    # Reshape the array to 2D (time x (body parts combined))
    reshaped_arr = arr.reshape(arr.shape[0], -1)

    # Find the flattened indices of the minimum values in the reshaped array
    flat_indices = cp.argmin(reshaped_arr, axis=1)

    # Convert flattened indices to 2D indices in the original n x n body parts grid
    n = arr.shape[1]  # Assuming the second and third dimensions are the same
    j_indices, k_indices = cp.divmod(flat_indices, n)

    return min_distances, (j_indices, k_indices)

