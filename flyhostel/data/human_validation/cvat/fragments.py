import logging
import numpy as np

logger=logging.getLogger(__name__)

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


def _max_fragment(fragments):
    fragments=np.asarray(fragments, dtype=np.float64)
    fragments=fragments[~np.isnan(fragments)]
    return fragments.max() if fragments.size else -1


def _follow_fragment(rows, x0, y0):
    """
    Walk `rows` (one fragment, sorted by frame_number) forward from (x0, y0),
    keeping one blob per frame: the one closest to the blob kept in the previous frame.

    Needed when the machine gave the same fragment id to two flies at the same time,
    so that only the tagged fly is followed. Returns the index labels of the kept rows
    and a {frame_number: (x, y)} dict of the kept positions.
    """
    frames=rows["frame_number"].values
    xy=rows[["x", "y"]].values.astype(np.float64)
    labels=rows.index.values

    kept=[]
    positions={}
    prev=np.array([x0, y0], dtype=np.float64)
    # boundaries of each frame's block of rows
    starts=np.flatnonzero(np.r_[True, frames[1:]!=frames[:-1]])
    ends=np.r_[starts[1:], len(frames)]
    for start, end in zip(starts, ends):
        if end-start==1:
            best=start
        else:
            best=start+np.argmin(((xy[start:end]-prev)**2).sum(axis=1))
        kept.append(labels[best])
        prev=xy[best]
        positions[frames[best]]=prev
    return kept, positions


def reassign_fragments(roi0_machine, roi0_human, reassignments, chunksize):
    """
    Apply NEW-FRAGMENT tags: from the tagged frame until the end of the chunk,
    the tagged fly's fragment gets a fresh id, both in the machine data and in the human data.

    This fixes the case where the machine assigned the same fragment id to two different flies,
    either one after the other (fly A, then fly B) or at the same time. Renaming the
    machine data too (not only the human-annotated frames) is what stops the identity
    a human gave to fly A from spreading through the fragment to fly B's blobs.

    Arguments
        * roi0_machine (pd.DataFrame): machine ROI_0 table with frame_number, in_frame_index, x, y, fragment
        * roi0_human (pd.DataFrame): human ROI_0 table with frame_number, x, y, fragment
        * reassignments (list): (frame_number, fragment, x, y) of each tag, where fragment is the
            machine fragment the tag overlapped and x, y is the center of the tag
        * chunksize (int)

    Returns
        roi0_machine, roi0_human: copies with the new fragment ids
    """
    roi0_machine=roi0_machine.copy()
    roi0_human=roi0_human.copy()
    if not reassignments:
        return roi0_machine, roi0_human

    machine_chunk=roi0_machine["frame_number"].values//chunksize
    human_chunk=roi0_human["frame_number"].values//chunksize
    # fragment ids as the machine gave them, to match each tag to its blob
    # even after an earlier tag has already renamed it
    orig_fragment=roi0_machine["fragment"].values.copy()

    for frame_number, fragment, x, y in sorted(reassignments, key=lambda r: r[0]):
        if fragment is None or np.isnan(fragment):
            logger.warning("NEW-FRAGMENT tag in frame %s does not overlap any machine fragment. Ignoring", frame_number)
            continue

        chunk=frame_number//chunksize
        in_chunk=machine_chunk==chunk

        # find the tagged blob and the fragment it carries now
        tagged=roi0_machine.loc[in_chunk & (roi0_machine["frame_number"].values==frame_number) & (orig_fragment==fragment)]
        if tagged.shape[0]==0:
            logger.warning("NEW-FRAGMENT tag in frame %s: fragment %s not found in machine data. Ignoring", frame_number, fragment)
            continue
        dist=(tagged["x"]-x)**2+(tagged["y"]-y)**2
        current_fragment=tagged.loc[dist.idxmin(), "fragment"]

        rows=roi0_machine.loc[
            in_chunk & (roi0_machine["frame_number"].values>=frame_number) & (roi0_machine["fragment"].values==current_fragment)
        ].sort_values(["frame_number", "in_frame_index"])
        kept, positions=_follow_fragment(rows, x, y)

        new_fragment=max(
            _max_fragment(roi0_machine["fragment"].values[in_chunk]),
            _max_fragment(roi0_human["fragment"].values[human_chunk==chunk]),
        )+1

        # human rows: rename those closest to the followed blob
        # (all of them, if no other blob shares the fragment in that frame)
        human_mask=(human_chunk==chunk) & (roi0_human["frame_number"].values>=frame_number) & (roi0_human["fragment"].values==current_fragment)
        others=rows.drop(index=kept)
        human_rename=[]
        for idx, row in roi0_human.loc[human_mask].iterrows():
            fn=row["frame_number"]
            if fn not in positions:
                human_rename.append(idx)
                continue
            other_xy=others.loc[others["frame_number"]==fn, ["x", "y"]].values
            point=np.array([row["x"], row["y"]], dtype=np.float64)
            d_kept=((positions[fn]-point)**2).sum()
            if other_xy.shape[0]==0 or d_kept<=((other_xy-point)**2).sum(axis=1).min():
                human_rename.append(idx)

        roi0_machine.loc[kept, "fragment"]=new_fragment
        roi0_human.loc[human_rename, "fragment"]=new_fragment

        logger.warning(
            "NEW-FRAGMENT: fragment %s in chunk %s becomes fragment %s from frame %s (%s machine rows, %s human rows)",
            current_fragment, chunk, new_fragment, frame_number, len(kept), len(human_rename)
        )

    return roi0_machine, roi0_human
