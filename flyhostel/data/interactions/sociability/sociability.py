import time
import itertools
import pickle
import os.path
import logging
import warnings
import traceback
from concurrent.futures import ProcessPoolExecutor

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from umap import UMAP
import numba
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
from pynndescent.distances import euclidean
from matplotlib.colors import ListedColormap
import colorcet as cc
from mplex import Grid
from flyhostel.data.interactions.classifier.inter_orientation import (
    sync_datasources
)
from flyhostel.data.pose.main import FlyHostelLoader
from flyhostel.data.pose.constants import chunksize as CHUNKSIZE
logger=logging.getLogger(__name__)


GET_FEATURES_BODYPARTS=["head", "thorax", "abdomen"]

def write_umap_tools(window_f):
    """
    Generate functions which may not take keyword arguments
    with the correct default value for the arguments
    """
    @numba.njit
    def flip(x, window_f=window_f):
        x = np.asarray(x)
        return np.concatenate((
            x[..., :window_f],
            x[..., window_f*4:],
            x[..., window_f:(window_f*4)]
        ))

    @numba.njit
    def my_dist(a, b, window_f=window_f):
        return min(euclidean(a, b), euclidean(a, flip(b, window_f=window_f)))

    return flip, my_dist


def interpolate_bp(imaginary_series):
    # Separate the real and imaginary parts and interpolate each one
    real_interp = imaginary_series.apply(np.real).interpolate()
    imag_interp = imaginary_series.apply(np.imag).interpolate()

    # Recombine into a complex series
    s_interp = real_interp + 1j * imag_interp
    return s_interp


def get_features(
        fly1: pd.DataFrame, fly2: pd.DataFrame,
        frame_indices: np.ndarray, pixel_to_mm: float,
        fps: float, window: float
    ):
    """Compute features for a pair of flies.
    https://www.biorxiv.org/content/10.1101/2024.11.25.624845v1.full.pdf
    https://github.com/NeLy-EPFL/Sociability_Learning/blob/e20721a9eb3b2f6f79f4ff7491845b241b824a4b/scripts/embedding.ipynb

    Parameters
    ----------
    fly1 : pd.DataFrame
        DataFrame with the position of the body parts of fly 1.
        Coordinates are encoded as complex numbers
        (real and imaginary parts are x- and y-coordinates, respectively).
        Must have same number of rows as fly2, and share the same tiemstamps.
    fly2 : pd.DataFrame
        DataFrame with the position of the body parts of fly 2.
        Coordinates are encoded as complex numbers
        (real and imaginary parts are x- and y-coordinates, respectively).
        Must have same number of rows as fly1, and share the same tiemstamps.
    frame_indices : np.ndarray
        Frame indices around which time windows are extracted.
        Reference to the rows of fly1 and fly2
        i.e. index=0 refers to the first row in fly1 and fly2
    pixel_to_mm: float
        mm / pixels (how many mm make up 1 pixel)
    fps: float
        how many datapoints per second
    window: float
        number of seconds around the frame index to be taken to capture the fly interaction

    Returns
    -------
    pd.DataFrame
        DataFrame with features for the pair of flies.
    """
    
    feature_names=["d", "vt1", "abs_vp1", "theta1", "vt2", "abs_vp2", "theta2"]

    window_f=int(window*fps)
    if window_f % 2 != 0:
        window_f+=1

    arange = np.arange(-window_f // 2, window_f // 2)

    pose1=fly1[GET_FEATURES_BODYPARTS]
    pose2=fly2[GET_FEATURES_BODYPARTS]


    # position of the flies
    p1 = np.mean(pose1.values, axis=1) * pixel_to_mm
    p2 = np.mean(pose2.values, axis=1) * pixel_to_mm

    eps = 0.01                     # or any suitably tiny value
    p1p2 = p2 - p1  # vector from fly1 to fly2
    dist = np.abs(p1p2)  # distance between flies

    # ---- protect zero-length vectors ------------------------------------------
    zero = dist < eps                       # boolean mask of the bad frames

    # replace the zero vectors with a tiny one pointing along +x
    # (1+0j) is arbitrary; it disappears after normalisation anyway
    p1p2[zero] = (eps + 0j)
    dist[zero]  = eps                       # keep the distance feature at 0 after the next line
    # ---------------------------------------------------------------------------

    p1p2 /= dist                            # safe normalisation

    # velocity of the flies in image coordinates
    v1 = gaussian_filter1d(p1, 2, order=1, mode="nearest") * fps
    v2 = gaussian_filter1d(p2, 2, order=1, mode="nearest") * fps

    # rotate velocities so that 1 + 0j points towards the other fly
    v1 /= p1p2
    v2 /= -p1p2

    # heading of the flies in image coordinates
    heading1 = pose1["head"].values * pixel_to_mm - p1
    heading2 = pose2["head"].values * pixel_to_mm - p2

    # rotate headings so that 1 + 0j points towards the other fly
    heading1 /= np.abs(heading1)
    heading2 /= np.abs(heading2)

    # convert complex numbers to angles and take absolute value
    theta1 = np.abs(np.angle(heading1 / p1p2))
    theta2 = np.abs(np.angle(heading2 / -p1p2))

    # interpolate theta values
    # they might still be nan if the head is found to be in the same place
    # as the computed position of the flies (which can occur in pose errors)
    theta1=pd.Series(theta1).interpolate().values
    theta2=pd.Series(theta2).interpolate().values
    

    # column-stack all features
    X = np.column_stack(
        [dist, v1.real, np.abs(v1.imag), theta1, v2.real, np.abs(v2.imag), theta2]
    )

    if np.isnan(X).sum()>0:
        print(np.where(np.isnan(X)))
        raise ValueError("Missing values detected")

    # exclude frame_indices close to the end of the timeseries
    frame_indices=frame_indices[frame_indices<(X.shape[0]-len(arange)/2)]
    
    # get time windows of length win_len starting from movement onsets
    X=X[frame_indices[:, None] + arange].transpose((0, 2, 1))

    frame_numbers=fly1["frame_number"].iloc[frame_indices]
    frame_numbers2=fly2["frame_number"].iloc[frame_indices]
    assert (frame_numbers.values==frame_numbers2.values).all()

    columns = pd.MultiIndex.from_tuples(
        itertools.product(feature_names, arange)
    )
    return pd.DataFrame(X.reshape((len(X), -1)), frame_numbers, columns)

def reproject(X, theta):
    """
    Project data onto new coordinate system
    The new coordinate system has one axis A theta degrees counter clockwise from current X
    and the other axis B is perpendicular to A, 90 degrees coutner clockwise
    """
    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(theta), np.sin(theta)],  # Project onto rotated X (V-axis)
        [-np.sin(theta), np.cos(theta)]  # Project onto rotated Y (perpendicular)
    ])
    
    # Project the velocity vectors
    Xp = X @ rotation_matrix.T
    return Xp

def z_normalize(X: np.ndarray):
    """Apply z-normalization to X in place.

    Parameters
    ----------
    X : np.ndarray
        Array to be z-normalized.
    """
    X -= X.mean()
    X /= X.std()
    return X

def rotate_embedding(Z):
    Z = Z - Z.mean(0)
    return Z @ PCA(n_components=2).fit(Z).components_.T


def get_features_all(loaders, window_s, min_time=None, max_time=None, timepoints=("frame_number",)):
    """
    Compute interaction features
    The features are computed for every interaction around the timepoints / milestones specified in timepoints
        Three are possible:
        * frame_number: Around the point of closest contact
        * first_frame: Around the first frame of the interaction (before which distance is > proximity threshold)
        * last_frame_number: Around the last frame of the interaction (after which distance is > proximity threshold)
    """
    indices=[]
    features=[]
    index=None

    experiments=list(set([loader.experiment for loader in loaders]))
    if len(experiments)==1: experiment=experiments[0]
    else: experiment="Mix"
    
    for loader1 in tqdm(loaders, desc=f"Analyzing interactions of {experiment}"):

        interactions=loader1.interaction.copy()

        if min_time is not None:
            interactions=interactions.loc[interactions["t"]>=min_time]
        if max_time is not None:
            interactions=interactions.loc[interactions["t"]<max_time]
            
        closest_distance=interactions.loc[interactions["distance_mm_min_flag"]].copy()

        for loader2 in loaders:
            if loader1==loader2:
                continue
            else:

                assert loader1.pose_complex is not None
                assert loader2.pose_complex is not None

                closest_distance_pair=closest_distance.loc[
                    closest_distance["nn"]==loader2.ids[0]
                ].sort_values("frame_number")
                pose_complex1, pose_complex2=sync_datasources(
                    loader1.pose_complex.copy(),
                    loader2.pose_complex.copy()
                )

                frame_indices_all=[]
                frame_numbers_all=[]

                closest_distance_pair_with_pose=closest_distance_pair.loc[
                    (closest_distance_pair["frame_number"] >= pose_complex1["frame_number"].iloc[0]) &
                    (closest_distance_pair["frame_number"] <= pose_complex1["frame_number"].iloc[-1])
                ]

                closest_distance_pair_without_pose=closest_distance_pair.loc[
                    (closest_distance_pair["frame_number"] < pose_complex1["frame_number"].iloc[0]) |
                    (closest_distance_pair["frame_number"] > pose_complex1["frame_number"].iloc[-1])
                ]
                if closest_distance_pair_without_pose.shape[0]>0:
                    logger.warning(
                        "%s interactions have no pose data, in chunks: %s",
                        closest_distance_pair_without_pose.shape[0],
                        sorted(list(set((closest_distance_pair_without_pose["frame_number"]//CHUNKSIZE).tolist())))
                    )


                print(f"Will load timepoints: {timepoints}")
                for timepoint in timepoints:
                    frame_numbers=closest_distance_pair_with_pose[timepoint]
                    first_frame_in_pose_dataset=pose_complex1["frame_number"].iloc[0]
                    assert first_frame_in_pose_dataset==pose_complex2["frame_number"].iloc[0]
                    
                    frames=(frame_numbers-first_frame_in_pose_dataset).values
                    # frames=np.where(pose_complex1["frame_number"].isin(frame_numbers).values)[0]
                    
                    if len(frames) != len(frame_numbers):
                        logger.warning("Some frames are missing in the pose")

                    frame_indices_all.append(frames)
                    frame_numbers_found=pose_complex1["frame_number"].iloc[frames]
                    frame_numbers_all.extend(frame_numbers_found)
                
                if len(frame_indices_all[0])==0:
                    logger.error("No interaction between %s and %s", loader1.ids[0], loader2.ids[0])
                    continue

                features_=get_features_multi(
                    pose_complex1, pose_complex2, frame_indices_all,
                    1/loader1.px_per_mm, fps=loader1.framerate,
                    window=window_s
                )

                index=closest_distance_pair_with_pose.loc[
                  closest_distance_pair_with_pose["frame_number"].isin(frame_numbers_all)
                ]

                if index.shape[0]==features_.shape[0]:
                    features.append(features_)
                    indices.append(index)
                else:
                    logger.error(
                        "index rows %s and features rows %s dont match in %s: %s vs %s",
                        index.shape[0], features_.shape[0], loader1.experiment, loader1.identity, loader2.identity
                    )

    if index is None or len(index)==0:
        return None, None
    
    index=pd.concat(indices, axis=0).reset_index(drop=True)
    df_dict={
      "features": pd.concat(features, axis=0).reset_index(drop=True)
    }
    assert df_dict["features"].shape[0]==index.shape[0]
    return df_dict, index

def get_features_multi(pose_complex1, pose_complex2, frame_indices_all, *args, **kwargs):
    features_all=[]
    for frame_indices in frame_indices_all:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            features=get_features(pose_complex1, pose_complex2, frame_indices, *args, **kwargs)        
        features_all.append(features.reset_index(drop=True))

    # Get unique feature names and positions from the first event
    feature_names = features_all[0].columns.get_level_values(0).unique()
    positions = features_all[0].columns.get_level_values(1).unique()
    num_events = len(features_all)

    # Concatenate across all features
    df = pd.concat(features_all, axis=1)

    # Create a MultiIndex: (feature, position, timepoint)
    # Each event is a 'timepoint'
    arrays = [
        list(feature_names.repeat(len(positions))) * num_events,
        list(positions.tolist() * (num_events*len(feature_names))),
        [str(i) for i in range(num_events) for _ in range(len(feature_names)*len(positions))]
    ]
    df.columns = pd.MultiIndex.from_arrays(arrays, names=('feature', 'position', 'timepoint'))

    return df


def load_data(loader, min_time=None, max_time=None, loaders_cache=None, identities=None):

    cache_file=f"{loaders_cache}/{loader.experiment}__{str(loader.identity).zfill(2)}.pkl"
    
    out=None
    if os.path.exists(cache_file):
        out=loader.load_from_cache(cache_file)
    if out is None:
        if min_time is not None and max_time is not None: assert min_time < max_time
        loader.load_centroid_data(min_time=min_time, max_time=max_time)
        loader.load_pose_data(min_time=min_time, max_time=max_time)
    else:
        loader=out

    loader.load_interaction_data(proximity_threshold=5, identities=identities)

    loader.add_centroid_data_to_pose()
    loader.project_to_absolute_coords_all(GET_FEATURES_BODYPARTS)
    loader.generate_pose_complex(loader.pose_absolute, GET_FEATURES_BODYPARTS)

    first_centroid_frame=loader.dt["frame_number"].iloc[0]
    first_pose_frame=loader.pose["frame_number"].iloc[0]

    print(f"First centroid frame {first_centroid_frame}. First pose frame {first_pose_frame}")

    if "1X" not in loader.basedir:
        assert loader.interaction is not None

        # to be added to load_interaction_data
        loader.interaction["distance_mm_min_flag"]=loader.interaction["distance_mm"]==loader.interaction["distance_mm_min"]

        # in case there is more than 1 frame with a fly-fly distance equal to the minimum, keep the first one
        # this is particularly relevant for interactions that are not properly validated
        # and because of that have both identities projected on to the same fly blob (therefore having distance_mm_min=0)
        first_closest_index=loader.interaction.loc[
            loader.interaction["distance_mm_min_flag"]==True,
            ["id", "nn", "interaction", "frame_number", "distance_mm_min_flag"]
        ].groupby([
            "id", "nn", "interaction"
        ]).first().reset_index()
        
        loader.interaction=loader.interaction.drop("distance_mm_min_flag", axis=1).merge(
            first_closest_index,
            on=["id", "nn", "interaction", "frame_number"],
            how="left"
        )

        loader.interaction["distance_mm_min_flag"]=np.where(
            loader.interaction["distance_mm_min_flag"].isna(),
            False,
            loader.interaction["distance_mm_min_flag"]
        )

        # overlap = fraction of the interaction where the two identities are projected on to the same fly by mistake
        loader.interaction["overlap"]=loader.interaction["distance_mm"]==0
        overlap_index=loader.interaction.groupby(["id", "nn", "interaction"]).agg({"overlap": np.mean}).reset_index()
        loader.interaction=loader.interaction.merge(
            overlap_index,
            on=["id", "nn", "interaction"],
            how="left"
        )

        closest_distance_count=loader.interaction.groupby(["id", "nn", "interaction"]).agg({
            "distance_mm_min_flag": np.sum
        }).rename({"distance_mm_min_flag": "count"}, axis=1)
        assert (closest_distance_count["count"]==1).all()


    # if not os.path.exists(cache_file):
    #     os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    #     with open(cache_file, "wb") as handle:
    #         pickle.dump(loader, handle)
    return loader

def load_data_all(experiment, identities, min_time=None, max_time=None, max_workers=1, **kwargs):
    """
    
    Arguments:
        max_workers (int): How many parallel processes launched with ProcessPoolExecutor to run simultaneously
            Each process loads the data of one animal, provided it has been cached already

    """
    loaders=[]

    try:
        for identity in identities:
            loaders.append(FlyHostelLoader(experiment=experiment, identity=identity))
    except Exception as error:
        logger.error(error)
        return []

    new_loaders=[]
    if max_workers==1:
        for loader in tqdm(loaders, desc=f"Loading data from {experiment}"):
            try:
                new_loaders.append(load_data(loader, min_time=min_time, max_time=max_time, **kwargs))
            except Exception as error:
                logger.error(error)
                pass
    else:
        # Parallel loading using processes
        min_times=[min_time, ]*len(identities)
        max_times=[max_time,]*len(identities)
        args=[]
        for kw in kwargs.keys():
            args.append([kwargs[kw], ]*len(identities))
        
        before=time.time()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            new_loaders = list(executor.map(load_data, loaders, min_times, max_times, *args))
        after=time.time()
        print(f"sociability.load_data_all done in {after-before} seconds")


    loaders=new_loaders
    for loader in loaders:
        for bp in GET_FEATURES_BODYPARTS:
            logger.info("%s - Interpolating %s", loader, bp)
            loader.pose_complex[bp]=interpolate_bp(loader.pose_complex[bp])
            # if loader.pose_complex[bp].isna().sum()!=0:
            assert loader.pose_complex[bp].isna().sum()==0


    return loaders

def analyze_experiment(loaders, window_s, *args, **kwargs):

    df, index=get_features_all(loaders, window_s, *args, **kwargs)
    assert df is not None, "Cannot compute interaction features"

    store_index=loaders[0].store_index
    min_time=loaders[0].dt["t"].min()
    max_time=loaders[0].dt["t"].max()

    index["keep"]=True

    index=index.merge(
        store_index[["frame_number", "t"]].rename({
            "t": "last_t",
            "frame_number": "last_frame_number"
        }, axis=1),
        on=["last_frame_number"], how="left"
    ).merge(
        store_index[["frame_number", "t"]].rename({
            "t": "first_t",
            "frame_number": "first_frame"
        }, axis=1),
        on=["first_frame"], how="left"
    )

    interactions_within_zt_limits=(index["first_t"]-window_s>=min_time)&(index["last_t"]+window_s<=max_time)
    index.loc[
        interactions_within_zt_limits,
        "keep"
    ]=False
    df["features"]=df["features"].iloc[np.where(index["keep"])]
    index=index.loc[index["keep"]]

    is_missing=np.isnan(df["features"].values).any(axis=1)
    assert is_missing.sum()==0, f"Features are missing in {is_mising.sum()} instances"
    features=df["features"]
    return features, index


def order_clusters(labels, new_order):
    # manually reorder clusters
    order = np.argsort(
        new_order
    )
        
    new_labels = np.zeros_like(labels)
    labels_u=np.unique(labels)
    for lab in labels:
        new_labels[labels == lab] = order[np.where(labels_u==lab)[0]]
    
    return new_labels


def visualize_clusters(Z, labels):

    if min(labels)==0:
        labels+=1

    labels_u=sorted(np.unique(labels).tolist())
    coms = np.array([Z[labels == k].mean(0) for k in labels_u])
    ax = Grid(100).item()
    ax.axis("off")
    ax.scatter(Z[:,0],Z[:,1], s=.1)
    ax.set_xlim(coms[:, 0].min(), coms[:, 0].max())
    ax.set_ylim(coms[:, 1].min(), coms[:, 1].max())
    ax.set_aspect("equal")
    
    for i, (x, y) in enumerate(coms):
        ax.text(x, y, labels_u[i], ha="center", va="center", fontsize=5)

def fit_symmetry(X, Z, labels, clustering, window_f):
    """
    Flip focal and side fly if the flip is closer to the centroid of its cluster
    """

    flip, _ = write_umap_tools(window_f)
    labels_u=sorted(np.unique(labels).tolist())
    need_flip = np.zeros(len(X), dtype=bool)
    
    
    for k in labels_u:
        idx = np.where(labels == k)[0]
        Xk = X[idx].copy()
        Zk = Z[idx]
        centroid_id = np.argmin(np.linalg.norm(Zk - clustering.cluster_centers_[k], axis=1))
        centroid = Xk[centroid_id]
    
        for i, x in enumerate(Xk):
            if euclidean(centroid, x) < euclidean(centroid, flip(x)):
                need_flip[idx[i]] = False
                Xk[i] = x
            else:
                need_flip[idx[i]] = True
                Xk[i] = flip(x)
    
        mean = Xk.mean(0)
    
        if mean[window_f:window_f*2].mean() < mean[window_f*4:window_f*5].mean():
            need_flip[idx] = ~need_flip[idx]
    return need_flip


def process_z(X, Z, n_clusters = 20, window_f=150):
    Z = rotate_embedding(Z) * (-1, 1)
    kmeans = KMeans(n_clusters, random_state=0)
    labels = kmeans.fit_predict(Z)
    labels+=1
    visualize_clusters(Z, labels)
    
    order=np.arange(n_clusters)
    assert (np.unique(order) == np.arange(n_clusters)).all()

    labels=order_clusters(labels, order)
    need_flip=fit_symmetry(X, Z, labels, kmeans, window_f=window_f)
    labels+=1
    return labels, need_flip, kmeans
    

def generate_s2(F, t, feature_names, clusters, output_path):
    """
    Generate suppl. fig 2 from https://www.biorxiv.org/content/10.1101/2024.11.25.624845v1.full.pdf

    The figure describes the mean and the dynamics of each of the features used to parametrize the interactions
    for each cluster, as a straightforward way to understand what differentiates each cluster from each other

    Arguments:
    --------------
        F (np.ndarray): Parametrization of interactions. Has 3 dimensions
           1. interactions: contains a vector of values of length #features x #timepoints (150*7=1050)
           2. features: one for every parameter extracted from the interaction (7)
           3. timepoints: one for every frame describing the interaction (150)
        t (np.ndarray): 1D vector of length equal to the 3rd axis of F
            containing the time point at which each frame was sampled,
            relative to the center of the interaction (point of closest contact).
        feature_names (iterable): Name of the parameters describing the interaction. Must have length equal to 2nd axis of F
        window_f (int): Number of frames describing the interaction. Equal to the size of the 3rd axis and the length of t
        clusters (iterable): Identifier of the group of interactios that each interaction belongs to. Length equal to size of first axis of F
        output_path (str): Path where to save the figure
    
    """
    FEATURE_ORDER = [0, 1, 4, 2, 5, 3, 6]
    clusters_u=sorted(list(set(clusters)))
    
    n_clusters=len(clusters_u)
    
    palette = ListedColormap(cc.rainbow)(np.linspace(0, 1, n_clusters))
    g = Grid((16, 20), (len(feature_names), n_clusters), sharey="row", space=(2, 6), dpi=144)
    g[:, :].set_visible_sides("")
    g[:, 0].set_visible_sides("l")
    ylabels = feature_names
    ts=[-.5, 0.5]
    v_lim=20
    d_lim=10
    ylims=[
        (0, d_lim),
        (-v_lim, v_lim),
        (-v_lim, v_lim),
        (0, v_lim),
        (0, v_lim),
        (0, np.pi),
        (0, np.pi),
    ]

    for cluster_name in tqdm(clusters_u):
        k=cluster_name-min(clusters_u)
        Fk = F[clusters == cluster_name]
        c = palette[k]

        for j in range(len(feature_names)):
            i = FEATURE_ORDER[j]
            # average over all bouts assigned to this cluster
            # at every timepoint
            feature = Fk[:, i]
    
            if k == 0:
                ax = g[j, :].make_ax(sharey=True)
                ax.axhline(0, color="k", ls="--", clip_on=False)
                ax.axis("on")
                ax.set_visible_sides("")
                label=ylabels[i]
                ax.add_text(0, 0.5, label, ha="c", va="c", transform="a", pad=(-23, 0))

            ax = g.axs[j, k]

            ax.set_xlim(t[0], t[-1])
            ax.vlines(ts, ymin=ylims[j][0], ymax=ylims[j][1])
            ax.set_ylim(*ylims[j])
            if "theta" in feature:
                ax.set_yticks(ylims[j], labels=["0", r"$\pi$"])
            else:
                ax.set_yticks(ylims[j])
            ax.plot(t, feature.mean(0), c=c, alpha=1, lw=1, clip_on=False)
            mean = feature.mean(0)
            std = feature.std(0)
    
            ax.fill_between(
                t, mean - std, mean + std, color=c, alpha=0.5, lw=0, clip_on=False
            )
            if i == 0:
                ax.set_title(f"{k+1}", color="k")

    if output_path is not None:
        print(f"Saving ---> {output_path}")
        g.savefig(output_path, transparent=True)
    return g


def analyze_interactions(features, window_f, run_name="run", n_clusters=20, n_neighbors=100, random_state=0):
    X=features.values.copy()

    flip, my_dist = write_umap_tools(window_f)
    # normalize within feature and within feature pair for focal and side animal
    for i in range(0, X.shape[1], window_f):
        z_normalize(X[:, i : i + window_f])
    z_normalize(X[:, :window_f])
    z_normalize(X[:, [*range(window_f, window_f*2), *range(window_f*4, window_f*5)]])
    z_normalize(X[:, [*range(window_f*2, window_f*3), *range(window_f*5, window_f*6)]])
    z_normalize(X[:, [*range(window_f*3, window_f*4), *range(window_f*6, window_f*7)]])

    # dr
    umap = UMAP(n_components=2, n_neighbors=n_neighbors, metric=my_dist, random_state=random_state)
    Z = umap.fit_transform(X)
    
    labels, need_flip, kmeans =process_z(X, Z, n_clusters=n_clusters, window_f=window_f)
    
    f = features.columns.get_level_values(0).unique() # 7 features
    F = features.copy() # all interactions x (features*timepoints)
    F = np.array([flip(f) if nf else f for f, nf in zip(F.values, need_flip)])
    F = F.reshape((F.shape[0], len(f), -1))
    # F has shape: interactions * 7 * timepoints   
    return umap, Z, F, labels, kmeans, need_flip


def process_experiment(experiment, identities, *args, window_s=1, cache=True, timepoints=("frame_number", ), **kwargs):
    """
    Describe interactions between animals in a single experiment
    
    window_s = Duration for which to compute features. Window is centered around each timepoint
    """

    try:
        loaders=load_data_all(experiment, identities, *args, **kwargs)
        features, index=analyze_experiment(loaders, window_s=window_s, timepoints=timepoints)
        index["experiment"]=experiment
        feature_names=features.columns.get_level_values(0).unique()
        features=pd.concat([
            features.loc[
                :,
                pd.IndexSlice[feat, :, :]
            ] for feat in feature_names
        ], axis=1)

    except Exception as error:
        logger.error("Cannot process experiment %s", experiment)
        logger.error(traceback.print_exc())
        logger.error(error)
        features=None
        index=None
        loaders=[]

    return experiment, (features, index), loaders