import math

import pandas as pd
import cudf
from idtrackerai_app.cli.utils.overlap import propagate_identities
from flyhostel.data.pose.constants import chunksize


def euclidean_distance(centroid1, centroid2):
    return ((centroid1-centroid2)**2).sum(axis=1)**0.5

def annotate_identity(data, number_of_animals):

    data=cudf.DataFrame(data)

    lid_table=cudf.concat([
        data[["chunk", "local_identity", "x", "y", "frame_number", "class_name", "modified"]].groupby(["chunk","local_identity"]).first().reset_index(),
        data[["chunk", "local_identity", "x", "y", "frame_number", "class_name", "modified"]].groupby(["chunk","local_identity"]).last().reset_index(),
    ], axis=0).sort_values(["frame_number", "local_identity"])
    lid_table["frame_idx"]=lid_table["frame_number"]%chunksize
    chunks=sorted(lid_table["chunk"].to_pandas().unique())

    identity_table=[]
    for chunk in chunks[:-1]:

        before = lid_table.loc[
            (lid_table["chunk"]==chunk) & (lid_table["frame_idx"]==44999)
        ]
        after = lid_table.loc[
            (lid_table["chunk"]==chunk+1) & (lid_table["frame_idx"]==0)
        ]
        
        for lid in before["local_identity"].to_pandas().unique():
            animal=before.loc[before["local_identity"]==lid]
            min_distance=math.inf
            selected_lid=None
            for i, lid_after in enumerate(after["local_identity"].to_pandas().unique()):
                next_animal=after.loc[after["local_identity"]==lid_after]
                if next_animal.shape[0]>1:
                    raise ValueError(f">1 animal found with local identity {lid_after} in chunk {chunk+1}")
       
                elif next_animal.shape[0]==0:
                    raise ValueError(f"0 animals found with local identity {lid_after} in chunk {chunk+1}")
        
                distance=euclidean_distance(
                    animal[["x", "y"]].values,
                    next_animal[["x", "y"]].values
                ).item()
                if distance < min_distance:
                    min_distance=distance
                    selected_lid=lid_after

            identity_table.append((chunk.item(), lid.item(), selected_lid.item(), min_distance))

    identity_table=pd.DataFrame.from_records(identity_table, columns=["chunk", "local_identity", "local_identity_after", "distance"])
    identity_table["is_inferred"]=False
    identity_table=propagate_identities(identity_table, chunks=chunks, ref_chunk=chunks[0], number_of_animals=number_of_animals, strict=True)
    data=data.merge(cudf.DataFrame(identity_table[["chunk", "local_identity", "identity"]]), on=["chunk", "local_identity"]).sort_values(["frame_number", "identity"]).to_pandas()
    return data