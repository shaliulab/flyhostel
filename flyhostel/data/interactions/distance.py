import itertools
import numpy as np
import cupy as cp
import pandas as pd
import codetiming, time
import logging

logger = logging.getLogger(__name__)

useGPU=True

def euclidean_distance(agent1, agent2, impl=np):
    return impl.sqrt(impl.sum(((agent1 - agent2)**2), axis=1))


def compute_distance_matrix(dt, use_gpu=None):
    """
    Compute euclidean distance between ALL agents in the dataset

    Arguments:

        dt (pd.DataFrame): Dataset of agent coordinates over time.
            Must contain columns id (with agent identities), frame_number (with timestamps),
            centroid_x and centroid_y with the x and y coordinates of the agent respectively
        use_gpu (bool): Whether to use the GPU for computations (using cupy) or not (using numpy's CPU implementation)


    Return:

        distance_matrix (np.array): The distance between the ith and jth agents at timestamp t is stored
            in the t, i, j position. Positions where i and j are the same are not included, therefore the shape of this array is
            n_timestamps x n_agents x n_agents - 1
            Examples:
                distance between 4 and 6th agent at time t is stored in t, 3, 4.
                the 3 represents the 4th agent
                the 4 represents the 5th agent that the 4th agent was compared against (which is the 6th global agent)
                
                distance between 1st and 2nd agents at time t is stored in t, 0, 0.
                the first 0 represents the 1st agent
                the second 0 represents the 1st agent that the 1st agent was compared against (which is the 2nd global agent)

        identities (iterable): Sorted list of the found agents in the dataset
        frame_number (iterable): Sorted list of the unique timestamps in the dataset
    """
    if use_gpu is not None:
        useGPU = use_gpu
    
    if useGPU:
        impl=cp
    else:
        impl=np

    df = dt[["id", "centroid_x", "centroid_y", "frame_number"]]
    df = df.sort_values(by=['id', 'frame_number'])

    # Pivot the dataframe
    x_pivot = df.pivot(index='id', columns='frame_number', values='centroid_x')
    y_pivot = df.pivot(index='id', columns='frame_number', values='centroid_y')

    identities=x_pivot.index.values.tolist()
    # Stack x and y values and reshape to get the desired shape
    result=(np.stack((x_pivot.values, y_pivot.values), axis=-1) * 100).astype(np.int64)
    result_gpu = cp.array(result) # shape agents x timestamps x 2
    number_of_animals=len(identities)

    pairs=[]
    distances=[]
    for i, j in itertools.combinations(np.arange(number_of_animals), 2):
        distances.append(
            euclidean_distance(
                result_gpu[i, :, :],
                result_gpu[j, :,:],
                impl
            )/100
        )
        pairs.append((
            identities[i],
            identities[j],
        ))

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

    frame_number=x_pivot.columns.values

    return distance_matrix, identities, frame_number


def compute_distance_matrix_bodyparts(interactions, pose, impl, bodyparts, precision=100):
    """
    Compute distance between bodyparts of two already close animals

    Arguments:
        interactions (pd.DataFrame): Dataset of close animal pairs with columns id, nn, frame_number,
            where id is a focal agent, nn is the nearest agent to the focal (nearest neighbor) and frame_number
            refers to the timestamp

        pose (pd.DataFrame): Dataset of bodypart coordinates for agents in the first dataset, with columns
            id, frame_number and foo_x and foo_y where for every foo bodypart

        impl: A reference to either numpy or cupy to use either the CPU or GPU for calculations
        bodyparts (list): A list of bodyparts to use to compute distances between animals. All bodyparts listed here must
           have a corresponding pair of _x and _y columns in the pose dataset 

    Returns:
        distance_matrix (np.array): Distance between any possible pair of bodyparts for every pair of agents in interactions
            The shape of this array is:
                number of rows in interactions (number of interactions) x number of possible bodypart pairs (for 6 bodyparts = 5+4+..+2+1=15)
        bp_pairs (np.array): Pairs of bodyparts, should have length equal to number of columns in distance_matrix
    """
    # combine information about the distance between centroids (interactions)
    # and the coordinates of the bps
    
    print("Merging pose and animal distance info")
    df1 = pd.merge(interactions, pose, how="left", left_on=["id","frame_number"], right_on=["id","frame_number"])
    df2 = pd.merge(interactions.drop("id", axis=1), pose, how="left", left_on=["nn","frame_number"], right_on=["id","frame_number"])

    assert all(df1["nn"] == df2["id"])

    x_columns=[f"{bp}_x" for bp in bodyparts]
    y_columns=[f"{bp}_y" for bp in bodyparts]
    x_coord = np.stack([
        df1[x_columns].values, df2[x_columns].values
    ])
    y_coord = np.stack([
        df1[y_columns].values, df2[y_columns].values
    ])

    coords=np.stack([x_coord, y_coord], axis=3)
    coords_cpu=np.round(coords*precision).astype(np.int64)

    coords=impl.array(coords_cpu)


    # coords shape = 2 x timetamps x bodyparts x 2
    # the first 2 corresponds to the pair of agents
    # second 2 corresponds to the x and y (space dimensionality)

    distances = []
    bp_pairs=[]
    with codetiming.Timer(text="Computing distances spent: {:.2f} seconds"):
        for i, (bp1, bp2) in enumerate(itertools.combinations(np.arange(len(bodyparts)), 2)):
            before=time.time()
            distance = euclidean_distance(coords[0, :, bp1, :], coords[1, :, bp2, :])
            distances.append(distance)
            bp_pairs.append((bp1, bp2))
            after=time.time()
            logger.debug(f"Loop {i}: {after-before} seconds")

    with codetiming.Timer(text="Stacking distances spent: {:.2f} seconds"):
        distance_matrix = impl.stack(distances, axis=1)

    bp_pairs=impl.array(bp_pairs)
    return distance_matrix, bp_pairs