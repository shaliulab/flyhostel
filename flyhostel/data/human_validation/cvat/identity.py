# cvat.identity
import math
import logging
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import cudf
from idtrackerai_app.cli.utils.overlap import propagate_identities
from flyhostel.utils import establish_dataframe_framework
from .courtship import remove_courtship_identities_from_local_identity_table
from .make_identity_table import make_identity_table
logger=logging.getLogger(__name__)



def make_local_identity_table(data, chunksize):
    xf=establish_dataframe_framework(data)

    data=xf.DataFrame(data.drop("identity", axis=1, errors="ignore"))
    
    first_frame=data[["chunk", "local_identity", "x", "y", "frame_number", "class_name", "modified"]].groupby(["chunk","local_identity"]).first().reset_index()
    last_frame=data[["chunk", "local_identity", "x", "y", "frame_number", "class_name", "modified"]].groupby(["chunk","local_identity"]).last().reset_index()
    first_frame["position"]="first"
    last_frame["position"]="last"

    lid_table=xf.concat([
        first_frame, last_frame
    ], axis=0).sort_values(["frame_number", "local_identity"])
    lid_table["frame_idx"]=lid_table["frame_number"]%chunksize
    return lid_table
            
def annotate_identity(data, number_of_animals, chunksize, debug=False,
                     annotated_table=None, verbose=True,
                     all_intervals_engaged_labels=None, **kwargs):
    """
    Generate the identity track for each animal in a dataset

    Given the local identity assigned to each animal in the first chunk, assign its value to all instances of the same
    animal throughout the experiment as a new attribute of the animal called identity

    The animal in the next chunk is selected by minimising the inter-animal distance between one animal of the last frame of the previous chunk
    and all animals in the first frame of the next chunk. The animal that minimises that distance is the same animal
    """

    xf=establish_dataframe_framework(data)
    data=xf.DataFrame(data.drop("identity", axis=1, errors="ignore"))
    if cudf is not None and xf is cudf:
        data_pandas=data.to_pandas()
    else:
        data_pandas=data

    lid_table=make_local_identity_table(data_pandas, chunksize)
    if number_of_animals>1:
        lid_table=lid_table.loc[lid_table["local_identity"]!=0]

    broken_tracks=lid_table.loc[~lid_table["frame_idx"].isin([0, chunksize-1])]
    # this can happen if a fly changes fragment
    # and regains the wrong local id in the process
    for _, track in broken_tracks.iterrows():
        info=f'Frame number: {int(track["frame_number"])} Local identity: {track["local_identity"]}. Position: {track["position"]}'
        if verbose:
            logger.warning(f"Track broken {info}")

    chunks=sorted(lid_table["chunk"].unique())

    
    lid_table = remove_courtship_identities_from_local_identity_table(
        lid_table, chunksize=chunksize, **kwargs,
    )
    lid_table.to_csv("local_identity_table.csv")

    identity_table = make_identity_table(
        lid_table, annotated_table, chunks,
        all_intervals_engaged_labels=all_intervals_engaged_labels,
        verbose=verbose, debug=debug,
    )

    counts=identity_table.value_counts(["chunk", "local_identity_after", "chunk_after"]).reset_index(name="count")
    error_df=counts.query("count>1")
    if error_df.shape[0]>0:
        import ipdb; ipdb.set_trace()
        raise ValueError("Local identity after is repeated. See identity_table.csv")

    logger.debug("Propagate identities")
    
    ref_chunk=chunks[0]
    print(f"Reference chunk = {ref_chunk}")
    identity_table=propagate_identities(
        identity_table, chunks=chunks, ref_chunk=ref_chunk,
        number_of_animals=number_of_animals, strict=True
    )
    logger.debug("Done")

    logger.debug("Merge identity annotation")
    data=data.to_pandas().merge(
        identity_table[["chunk", "local_identity", "identity"]],
        on=["chunk", "local_identity"]
    ).sort_values([
        "frame_number", "identity"
    ])

    return data