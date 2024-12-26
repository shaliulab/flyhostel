import math
import logging

import pandas as pd
import cudf
from idtrackerai_app.cli.utils.overlap import propagate_identities
from flyhostel.data.pose.constants import chunksize as CHUNKSIZE
from flyhostel.utils import establish_dataframe_framework

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
    lids=lids.unique()

    for i, lid_after in enumerate(lids):
        next_animal=after.loc[after["local_identity"]==lid_after]
        if next_animal.shape[0]>1:
            # import ipdb; ipdb.set_trace()
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



def make_identity_table(lid_table, chunks):
    identity_table=[]
    for chunk in chunks[:-1]:
        used_local_identity_after=set([])

        before = lid_table.loc[
            (lid_table["chunk"]==chunk) & (lid_table["frame_idx"]==(CHUNKSIZE-1))
        ]
        after = lid_table.loc[
            (lid_table["chunk"]==chunk+1) & (lid_table["frame_idx"]==0)
        ]
        
        lids=before["local_identity"]
        if isinstance(lids, cudf.Series):
            lids=lids.to_pandas()
        lids=lids.unique()
        with open("identity_table.log", "w") as log:
            for lid in lids:
                local_identity, min_distance=match_animals_between_chunks_by_distance(before, after, lid, chunk, log)
                if local_identity in used_local_identity_after:
                    logger.warning("%s already used in chunk %s", local_identity, chunk)
                    log.write(f"{local_identity} already used in chunk {chunk}\n")
                else:
                    used_local_identity_after.add(local_identity.item())
                
                identity_table.append((chunk.item(), lid.item(), local_identity.item(), min_distance))

    identity_table=pd.DataFrame.from_records(identity_table, columns=["chunk", "local_identity", "local_identity_after", "distance"])
    identity_table["is_inferred"]=False
    return identity_table
            
def annotate_identity(data, number_of_animals):
    """
    Generate the identity track for each animal in a dataset

    Given the local identity assigned to each animal in the first chunk, assign its value to all instances of the same
    animal throughout the experiment as a new attribute of the animal called identity

    The animal in the next chunk is selected by minimising the inter-animal distance between one animal of the last frame of the previous chunk
    and all animals in the first frame of the next chunk. The animal that minimises that distance is the same animal
    """

    xf=establish_dataframe_framework(data)
    data=xf.DataFrame(data)

    lid_table=xf.concat([
        data[["chunk", "local_identity", "x", "y", "frame_number", "class_name", "modified"]].groupby(["chunk","local_identity"]).first().reset_index(),
        data[["chunk", "local_identity", "x", "y", "frame_number", "class_name", "modified"]].groupby(["chunk","local_identity"]).last().reset_index(),
    ], axis=0).sort_values(["frame_number", "local_identity"])
    lid_table["frame_idx"]=lid_table["frame_number"]%CHUNKSIZE
    
    broken_tracks=lid_table.loc[~lid_table["frame_idx"].isin([0, CHUNKSIZE-1])].to_pandas()
    for _, track in broken_tracks.iterrows():
        info=f'Frame number: {track["frame_number"]} Local identity: {track["local_identity"]}'
        logger.warning(f"Track broken {info}")

    chunks=sorted(lid_table["chunk"].to_pandas().unique())

    identity_table=make_identity_table(lid_table, chunks)
    identity_table.to_csv("identity_table.csv")
    
    logger.debug("Propagate identities")
    identity_table=propagate_identities(identity_table, chunks=chunks, ref_chunk=chunks[0], number_of_animals=number_of_animals, strict=True)
    logger.debug("Done")

    logger.debug("Merge identity annotation")
    data=data.to_pandas().merge(
        identity_table[["chunk", "local_identity", "identity"]],
        on=["chunk", "local_identity"]
    ).sort_values([
        "frame_number", "identity"
    ])
    logger.debug("Done")
    data=xf.DataFrame(data)
    return data