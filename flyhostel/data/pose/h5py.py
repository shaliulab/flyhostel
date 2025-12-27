import logging
import os
import time
import itertools
import h5py
import numpy as np
import pandas as pd

from flyhostel.data.pose.constants import bodyparts as BODYPARTS

logger = logging.getLogger(__name__)
time_counter=logging.getLogger("time_counter")

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


def load_pose_data_compiled(datasetnames, ids, lq_thresh, chunksize, files, stride=1, min_time=None, max_time=None, store_index=None):
    """
    Load dataset TODO
    """
    before_out=time.time()

    pose_list=[]
    h5s_pandas=[]
    index_pandas=[]

    for animal_id, datasetname in enumerate(datasetnames):
        this_animal_files=files[animal_id]

        if isinstance(this_animal_files, str):
            h5_file_raw=this_animal_files
            h5_file_filtered=h5_file_raw
        elif len(this_animal_files)==1:
            h5_file_raw=this_animal_files[0]
            h5_file_filtered=h5_file_raw
        else:
            h5_file_filtered, h5_file_raw=this_animal_files
        
        identity=int(os.path.splitext(os.path.basename(h5_file_filtered))[0].split("__")[1])

        if not os.path.exists(h5_file_filtered):
            print(f"{h5_file_filtered} not found")
            continue

        filehandle_raw=h5py.File(h5_file_raw)
        logger.debug("Opening %s", h5_file_filtered)
        try:
            filehandle=h5py.File(h5_file_filtered)
        except Exception as error:
            logger.error("Cannot open %s", h5_file_filtered)
            raise error
        
        before=time.time()
        first_chunk=int(os.path.basename(filehandle["files"][0].decode()).split(".")[0])
        first_frame_number=first_chunk*chunksize

        if min_time is not None:
            # select the first frame_number whose t is greater or equal than min time
            fn0=store_index.loc[store_index["t"]>=min_time, "frame_number"].iloc[0]
        else:
            fn0=0
        last_frame_number_available=filehandle["tracks"].shape[3]+first_frame_number
        if max_time is not None:
            # select the first frame number whose t is greater than max time
            fn1=store_index.loc[store_index["t"]>max_time, "frame_number"]
            if len(fn1)==0:
                fn1=last_frame_number_available
            else:
                fn1=fn1.iloc[0]
        else:
            fn1=last_frame_number_available

        do_warning=False
        last_frame_not_available=None
        if fn1 > last_frame_number_available:
            do_warning=True
            last_frame_not_available=fn1
            fn1=last_frame_number_available

        fn0=max(0, fn0-first_frame_number)
        fn1=max(1, fn1-first_frame_number)
        
        frame_numbers=np.arange(fn0, fn1, stride) + first_frame_number
        if do_warning:
            logger.warning("Requested interval is partially not available. Frames not available: (%s - %s)", frame_numbers[-1], last_frame_not_available)
            
        n_files=filehandle["files"].shape[0]
        n_chunks=filehandle["tracks"].shape[3]/chunksize
        if n_files > n_chunks:
            raise Exception(f"{n_files} files should contain {round(n_chunks, 5)} of data. Some chunk may be incomplete")

        pose=filehandle["tracks"][0, :, :, fn0:fn1:stride]
        after=time.time()
        time_counter.debug("Load pose coordinates in %s seconds", round(after-before, 1))
        scores=filehandle_raw["point_scores"][0, :, fn0:fn1:stride]
        after2=time.time()
        time_counter.debug("Load scores in %s seconds", round(after2-after, 1))
        bps = [bp.decode() for bp in filehandle["node_names"][:]]

        # chunks=[int(os.path.basename(path.decode()).split(".")[0]) for path in filehandle["files"]]
        local_identity=[int(os.path.basename(os.path.dirname(path.decode()))) for path in filehandle["files"]]
        local_identity=list(itertools.chain(*[[e for _ in range(chunksize)] for e in local_identity]))
        analysis_files=list(itertools.chain(*[[e for _ in range(chunksize)] for e in filehandle["files"]]))

        local_identity=local_identity[fn0:fn1:stride]
        analysis_files=analysis_files[fn0:fn1:stride]

        filehandle.close()
        try:
            filehandle_raw.close()
        except:
            pass

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
        index["animal"]=datasetname
        index["index"]=index["frame_number"]
        index["files"]=analysis_files
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
        time_counter.debug("Initialize pose pd.DataFrame from Python dictionary in %s seconds", round(after-before, 1))
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
        time_counter.debug("Annotate pose dataset time in %s seconds", round(after-before, 1))
        pose_df["t"]=pose_df["zt"]*3600
        del pose_df["zt"]
        pose_df.insert(0, "t", pose_df.pop("t"))
        pose_df.insert(0, "id", ids[animal_id])
        pose_list.append(pose_df)
        index_pandas.append(index)

    if len(pose_list) == 0:
        return [], [], []
    else:
        return pose_list, h5s_pandas, index_pandas
