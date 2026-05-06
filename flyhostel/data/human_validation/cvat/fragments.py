import numpy as np

def make_identity_tracks(ident, roi0, chunksize, include_crossings=False):
    if include_crossings:
        pass
    else:
        ident=ident.loc[~(ident["is_a_crossing"]) ]
    identity_tracks=ident[["frame_number", "in_frame_index", "local_identity", "is_a_crossing"]].merge(
        roi0[["frame_number", "in_frame_index", "fragment"]],
        on=["frame_number", "in_frame_index"],
        how="left"
    )
    identity_tracks["chunk"]=identity_tracks["frame_number"]//chunksize
    identity_tracks["frame_idx"]=identity_tracks["frame_number"]%chunksize
    identity_tracks["validated"]=True
    identity_tracks.sort_values(["frame_number", "fragment"], inplace=True)
    if include_crossings:
        pass
    else:
        identity_tracks=identity_tracks.loc[~identity_tracks["fragment"].isna()]
    return identity_tracks


def make_identity_singletons(ident, roi0, chunksize):
    identity_singletons=make_identity_tracks(ident, roi0, chunksize, include_crossings=True)
    identity_singletons.loc[identity_singletons["is_a_crossing"]==True, "fragment"]=np.nan
    identity_singletons=identity_singletons.loc[identity_singletons["fragment"].isna()]
    identity_singletons=identity_singletons.merge(
        roi0[["in_frame_index", "frame_number", "x", "y"]],
        on=["in_frame_index", "frame_number"],
        how="left"
    )
    identity_singletons["validated"]=2
    return identity_singletons
