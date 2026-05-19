import math
import logging
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import cudf
from idtrackerai_app.cli.utils.overlap import propagate_identities
from flyhostel.utils import establish_dataframe_framework
from .courtship import remove_courtship_identities_from_local_identity_table
logger=logging.getLogger(__name__)

def euclidean_distance(centroid1, centroid2):
    return ((centroid1-centroid2)**2).sum(axis=1)**0.5


def match_animals_between_chunks_by_distance(before, after, local_identity_before, chunk, log=None):
    animal=before.loc[before["local_identity"]==local_identity_before]
    min_distance=math.inf
    selected_lid=None
    lids=after["local_identity"]
    if isinstance(lids, cudf.Series):
        lids=lids.to_pandas()
    lids=lids.unique().tolist()

    for i, lid_after in enumerate(lids):
        next_animal=after.loc[after["local_identity"]==lid_after]
        if next_animal.shape[0]>1:
            raise ValueError(f"{next_animal.shape[0]} animals found with local identity {lid_after} in chunk {chunk+1}")

        elif next_animal.shape[0]==0:
            raise ValueError(f"0 animals found with local identity {lid_after} in chunk {chunk+1}")

        distance=euclidean_distance(
            animal[["x", "y"]].values,
            next_animal[["x", "y"]].values
        ).item()
        if distance < min_distance:
            
            min_distance=distance
            selected_lid=lid_after
    
    if log is not None:
        # log.write("%s - %s -> %s - %s\n".format(chunk, local_identity_before, chunk+1, selected_lid))
        log.write(f"{chunk} - {local_identity_before} -> {chunk+1} - {selected_lid}\n")

    # logger.debug("%s - %s -> %s - %s", chunk, local_identity_before, chunk+1, selected_lid)
    return selected_lid, min_distance



def ensure_continuity_of_table(table):

    local_identities=table["local_identity"].drop_duplicates().tolist()
    table.sort_values(["chunk", "local_identity"], inplace=True)

    all_chunks={}

    for local_identity_start in local_identities:
        local_identity=local_identity_start

        chunks=[
            table.loc[table["local_identity"]==local_identity, "chunk"].iloc[0]
        ]

        local_identity_chain=[
            local_identity_start,
            table.loc[
                (table["local_identity"]==local_identity) & (table["chunk"]==chunks[-1]),
                "local_identity_after"
            ].item()
        ]
        while True:
            next_chunk=table.loc[
                (table["local_identity"]==local_identity_chain[-1]) & (table["chunk"]==chunks[-1]),
                "chunk_after"
            ]
            if len(next_chunk)==1:
                next_local_identity=table.loc[
                    (table["local_identity"]==local_identity_chain[-1]) & (table["chunk"]==chunks[-1]),
                    "local_identity_after"
                ].item()
            
                next_chunk=next_chunk.item()
                chunks.extend(list(range(chunks[-1]+1, next_chunk+1)))
                local_identity_chain.append(next_local_identity)

            else:
                break

        all_chunks[local_identity]=np.array(chunks)

    number_of_chunks=table["chunk"].drop_duplicates().shape[0]


    for local_identity in all_chunks:
        test1=(np.diff(all_chunks[local_identity])==1).all()
        length=len(np.diff(all_chunks[local_identity]))

        test2=length==number_of_chunks

        if not test1:
            print("Is validation_lags.csv correct?")
            print(local_identity_chain)
            print(all_chunks[local_identity])
            
            raise Exception(f"local_identity {local_identity} -> {all_chunks[local_identity]}")
        if not test2:
            print("Is validation_lags.csv correct?")
            print(local_identity_chain)
            print(all_chunks[local_identity])
            raise Exception(f"local_identity len({local_identity})!={number_of_chunks} ({length})")

def make_identity_table(lid_table, annotated_table, chunks, verbose=False, debug=False):
    """

    Details:
        lid_table format

        Two rows per fly and chunk, one for the start and one for the end of the chunk
        Contains the x / y coordinates of the fly and its local_identity
        This is used to find which local_identity does the fly get in the next chunk, which
        gets stored in the column local_identity_after of the output table.
        Propagate identities uses that info to then "propagate" the same identity for the same fly through chunks

        NOTE: If the fly is involved in courtship during a chunk, the entries for the fly in that chunk are removed
        If the courtship starts in the chunk, the position "last" is removed
        If the courtship ends in the chunk, the position "first" is removed

        identity_table format
        One row per fly and chunk

    """
    identity_table=[]
    for chunk in tqdm(chunks[:-1]):
        used_local_identity_after=set([])

        before = lid_table.loc[
            (lid_table["chunk"]==chunk) & (lid_table["position"]=="last")
        ]
        after = lid_table.loc[
            (lid_table["chunk"]==chunk+1) & (lid_table["position"]=="first")
        ]
        
        lids=before["local_identity"]
        if isinstance(lids, cudf.Series):
            lids=lids.to_pandas()
        lids=lids.unique().tolist()
        with open("identity_table.log", "w") as log:
            for local_identity in lids:
                local_identity_after, min_distance=match_animals_between_chunks_by_distance(before, after, local_identity, chunk, log)
                if local_identity_after in used_local_identity_after:
                    logger.warning("%s already used in chunk %s", local_identity_after, chunk)
                    log.write(f"{local_identity_after} already used in chunk {chunk}\n")
                    print(before, after)
                    if debug:
                        import ipdb; ipdb.set_trace()
                else:
                    used_local_identity_after.add(local_identity_after)
                
                if verbose:
                    print(f"chunk {chunk} - {local_identity} > {local_identity_after}")
                identity_table.append((chunk.item(), local_identity, local_identity_after, min_distance))

    identity_table=pd.DataFrame.from_records(identity_table, columns=["chunk", "local_identity", "local_identity_after", "distance"])
    identity_table["chunk"]=identity_table["chunk"].astype(int)
    identity_table["is_inferred"]=False
    identity_table["chunk_after"]=identity_table["chunk"]+1


    identity_table["priority"]=2

    
    if annotated_table is not None:
        annotated_table["distance"]=np.nan
        annotated_table["is_inferred"]=False
        annotated_table["priority"]=np.inf

        identity_table=pd.concat([
            # so that annotations take preference
            annotated_table,
            identity_table
        ], axis=0)\
            .sort_values(["priority", "chunk", "local_identity"], ascending=True)
        

        dups=identity_table.duplicated(["chunk_after", "local_identity_after"])
        if dups.any():
            if verbose:
                for _, row in identity_table.loc[dups].iterrows():
                    print(identity_table.loc[
                        (identity_table["chunk_after"]==row["chunk_after"]) & (identity_table["local_identity_after"]==row["local_identity_after"])
                    ])
            identity_table=identity_table.loc[~dups]
            

        identity_table.sort_values(["chunk", "local_identity"], inplace=True)


    dups=identity_table.duplicated(["chunk_after", "local_identity_after"])

    identity_table.to_csv("identity_table.csv")

    ensure_continuity_of_table(identity_table)

    return identity_table


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
            
def annotate_identity(data, number_of_animals, chunksize, debug=False, annotated_table=None, verbose=True, **kwargs):
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

    
    lid_table=remove_courtship_identities_from_local_identity_table(lid_table, chunksize=chunksize, **kwargs)
    lid_table.to_csv("local_identity_table.csv")
    identity_table=make_identity_table(lid_table, annotated_table, chunks, verbose=verbose, debug=debug)

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