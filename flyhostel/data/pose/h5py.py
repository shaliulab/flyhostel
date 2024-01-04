import logging
import os
import time
import itertools
import h5py
import numpy as np
import pandas as pd

from flyhostel.data.bodyparts import bodyparts as BODYPARTS

logger = logging.getLogger(__name__)

def clean_bad_proboscis(h5s_pandas, threshold):
    """
    If the score of the proboscis is too low
    the position is ignored
    and instead is set to be on the head
    is_interpolated in this case becomes True
    """
    for i, h5 in enumerate(h5s_pandas):
        bad_quality_rows=(h5.loc[:, pd.IndexSlice[:, ["proboscis"], "likelihood"]] < threshold).values.flatten()
        h5.loc[bad_quality_rows, pd.IndexSlice[:, "proboscis", "x"]]=h5.loc[bad_quality_rows, pd.IndexSlice[:, "head", "x"]]
        h5.loc[bad_quality_rows, pd.IndexSlice[:, "proboscis", "y"]]=h5.loc[bad_quality_rows, pd.IndexSlice[:, "head", "y"]]
        h5.loc[bad_quality_rows, pd.IndexSlice[:, "proboscis", "likelihood"]]=h5.loc[bad_quality_rows, pd.IndexSlice[:, "head", "likelihood"]]
        h5.loc[bad_quality_rows, pd.IndexSlice[:, "proboscis", "is_interpolated"]]=True
        h5s_pandas[i]=h5
    
    return h5s_pandas


def simplify_columns(index, pose, id):

    bps=np.unique(pose.columns.get_level_values(1).values)
    pose=pose.loc[:, pd.IndexSlice[:, bps, ["x","y", "is_interpolated", "likelihood"]]]
    pose.columns=itertools.chain(*[[bp + "_x", bp + "_y", bp + "_is_interpolated", bp + "_likelihood"] for bp in bps])
    pose=pose[itertools.chain(*[[bp + "_x", bp + "_y", bp + "_is_interpolated", bp + "_likelihood"] for bp in BODYPARTS])]

    pose=pose.merge(index[["frame_number", "zt"]].set_index("frame_number"), left_index=True, right_index=True)
    pose["t"]=pose["zt"]*3600
    del pose["zt"]
    pose.insert(0, "t", pose.pop("t"))
    pose.insert(0, "id", id)
    return pose


def load_pose_data_processed(min_time, max_time, time_system, datasetnames, identities, lq_thresh):
    """
    Load dataset stored in MOTIONMAPPER_DATA
    """

    pose_list=[]
    h5s_pandas=[]
    index_pandas=[]

    for animal_id, d in enumerate(datasetnames):
        h5_file = '%s/%s_positions.h5' % (os.environ["MOTIONMAPPER_DATA"], d)

        if not os.path.exists(h5_file):
            print(f"{h5_file} not found")
            continue

        index = pd.read_hdf(h5_file, key="index")
        index["t"] = index["zt"]*3600

        keep_rows=np.where((index["t"] >= min_time) & (index["t"] < max_time))[0]
        first_row=keep_rows[0]
        last_row=keep_rows[-1]+1
        
        index=index.iloc[first_row:last_row]
        pose=pd.read_hdf(h5_file, key="pose", start=first_row, stop=last_row)
        pose=clean_bad_proboscis([pose], lq_thresh)[0]
        index["animal"]=d
        index["index"]=index["frame_number"]
        index.set_index("index", inplace=True)

        h5s_pandas.append(pose)

        pose_list.append(simplify_columns(index, pose, identities[animal_id]))
        index_pandas.append(index)
    
    if len(pose_list) == 0:
        return None
    else:
        return pose_list, h5s_pandas, index_pandas

def load_pose_data_compiled(datasetnames, identities, lq_thresh, stride=1):
    """
    Load dataset stored in POSE_DATA
    """
    before_out=time.time()

    pose_list=[]
    h5s_pandas=[]
    index_pandas=[]

    for animal_id, d in enumerate(datasetnames):
        h5_file = '%s/%s/%s.h5' % (os.environ["POSE_DATA"], d, d)
        identity=int(os.path.splitext(os.path.basename(h5_file))[0].split("__")[1])


        if not os.path.exists(h5_file):
            print(f"{h5_file} not found")
            continue


        logger.debug("Opening %s", h5_file)
        with h5py.File(h5_file) as filehandle:
            chunksize=int(filehandle["tracks"].shape[3] / filehandle["files"].shape[0])
            before=time.time()
            pose=filehandle["tracks"][0, :, :, ::stride]
            after=time.time()
            logger.debug(f"Load pose coordinates in {round(after-before, 1)} seconds")
            scores=filehandle["point_scores"][0, :, ::stride]
            after2=time.time()
            logger.debug(f"Load scores in {round(after2-after, 1)} seconds")
            bps = [bp.decode() for bp in filehandle["node_names"][:]]

            chunks=[int(os.path.basename(path.decode()).split(".")[0]) for path in filehandle["files"]]
            local_identity=[int(os.path.basename(os.path.dirname(path.decode()))) for path in filehandle["files"]]
            local_identity=list(itertools.chain(*[[e for _ in range(chunksize)] for e in local_identity]))
            files=list(itertools.chain(*[[e for _ in range(chunksize)] for e in filehandle["files"]]))
            local_identity=local_identity[::stride]
            
            files=files[::stride]

            frame_numbers=list(itertools.chain(*[np.arange(
                chunk*chunksize,
                (chunk+1)*chunksize,
                1
            ) for chunk in chunks]))
            frame_numbers=frame_numbers[::stride]


        index=pd.DataFrame({
            "frame_number": frame_numbers,
        })
        index["chunk"]=index["frame_number"]//chunksize
        index["frame_idx"]=index["frame_number"] % chunksize
        
        index["frame_time"]=np.nan
        index["t"]=np.nan
        index["zt"]=np.nan
        index["local_identity"]=local_identity
        index["identity"]=identity
        index["animal"]=d
        index["index"]=index["frame_number"]
        index["files"]=files
        index.set_index("index", inplace=True)

        data={}
        coordinates=["x", "y", "likelihood", "is_interpolated"]
        before=time.time()

        # Whether is interpolated or not is independent of the value
        # It will be set to True by downstream programs
        is_interpolated = [False for _ in range(pose.shape[2])]
        for i, bp in enumerate(bps):
            data[bp + "_x"]=pose[0, i, :]
            data[bp + "_y"]=pose[1, i, :]
            data[bp + "_likelihood"]=scores[i, :]
            data[bp + "_is_interpolated"]=is_interpolated
        after = time.time()

        del pose
        del scores
        del is_interpolated

        before=time.time()
        pose_df = pd.DataFrame(data)
        after=time.time()
        logger.debug("Initialize pd.DataFrame in %s seconds", round(after-before, 1))
        # STOP adding the pose_df with fancy multiindex to h5s_pandas to save memory
        # columns=pose_df.columns
        # multiindex_columns = pd.MultiIndex.from_product([["SLEAP"], bps, coordinates], names=['scorer', 'bodyparts', 'coordinates'])
        # pose_df.columns=multiindex_columns
        # h5s_pandas.append(pose_df.copy())
        # pose_df.columns=columns
        pose_df["frame_number"]=frame_numbers
        pose_df.set_index("frame_number", inplace=True)

        before=time.time()
        pose_df=pose_df.merge(
            index[["frame_number", "zt"]].set_index("frame_number"),
            left_index=True, right_index=True
        )
        after=time.time()
        logger.debug("Annotate pose dataset time in %s seconds", round(after-before, 1))
        pose_df["t"]=pose_df["zt"]*3600
        del pose_df["zt"]
        pose_df.insert(0, "t", pose_df.pop("t"))
        pose_df.insert(0, "id", identities[animal_id])
        pose_list.append(pose_df)
        index_pandas.append(index)
    
    after_out=time.time()
    # logger.debug("Load pose data in % seconds", round(after_out-before_out, 1))

    if len(pose_list) == 0:
        return None
    else:
        return pose_list, h5s_pandas, index_pandas
