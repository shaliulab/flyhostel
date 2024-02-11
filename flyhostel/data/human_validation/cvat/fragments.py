import numpy as np
from flyhostel.data.pose.constants import chunksize

def make_fragment_index(ident, roi0, include_crossings=False):
    if include_crossings:
        pass
    else:
        ident=ident.loc[~(ident["is_a_crossing"]) ]
    lid_fragment_index=ident[["frame_number", "in_frame_index", "local_identity", "is_a_crossing"]].merge(
        roi0[["frame_number", "in_frame_index", "fragment"]],
        on=["frame_number", "in_frame_index"],
        how="left"
    )
    lid_fragment_index["chunk"]=lid_fragment_index["frame_number"]//chunksize
    lid_fragment_index["frame_idx"]=lid_fragment_index["frame_number"]%chunksize
    lid_fragment_index["validated"]=True
    lid_fragment_index.sort_values(["frame_number", "fragment"], inplace=True)
    if include_crossings:
        pass
    else:
        lid_fragment_index=lid_fragment_index.loc[~lid_fragment_index["fragment"].isna()]
    return lid_fragment_index


def make_annotation_wo_fragment_index(ident, roi0):
    df=make_fragment_index(ident, roi0, include_crossings=True)
    df.loc[df["is_a_crossing"]==True, "fragment"]=np.nan
    lid_fragment_index_nofragm=df.loc[df["fragment"].isna()]
    lid_fragment_index_nofragm=lid_fragment_index_nofragm.merge(roi0[["in_frame_index", "frame_number", "x", "y"]], on=["in_frame_index", "frame_number"], how="left")
    return lid_fragment_index_nofragm
