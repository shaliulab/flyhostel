# cvat.make_identity_table.py
import logging
import math

from collections import namedtuple

import numpy as np
import pandas as pd
import cudf
from tqdm.auto import tqdm
from .helpers import ensure_continuity_of_table

Bridge = namedtuple("Bridge", ["chunk", "local_identity", "chunk_after", "local_identity_after"])

LINK_COLUMNS = ["chunk", "local_identity", "chunk_after", "local_identity_after"]

logger=logging.getLogger(__name__)


def euclidean_distance(centroid1, centroid2):
    return ((centroid1-centroid2)**2).sum(axis=1)**0.5

def _to_pandas(series):
    """cudf.Series -> pandas.Series; anything else is returned unchanged."""
    return series.to_pandas() if hasattr(series, "to_pandas") else series


def _local_identities(lid_table, chunk, position):
    rows = lid_table.loc[(lid_table["chunk"] == chunk) & (lid_table["position"] == position)]
    return rows, sorted(int(x) for x in _to_pandas(rows["local_identity"]).unique().tolist())


def _parse_bridges(annotated_table, chunks):
    """Annotated links spanning more than one chunk boundary (e.g. copulation)."""
    if annotated_table is None or len(annotated_table) == 0:
        return []
    chunk_set = set(map(int, chunks))
    bridges = []
    for c, l, ca, la in annotated_table[LINK_COLUMNS].itertuples(index=False, name=None):
        bridge = Bridge(int(c), int(l), int(ca), int(la))
        if bridge.chunk_after <= bridge.chunk:
            raise ValueError(f"Annotated link does not move forward in time: {bridge}")
        if bridge.chunk not in chunk_set or bridge.chunk_after not in chunk_set:
            logger.warning("Annotated link %s refers to a chunk outside `chunks`", bridge)
        bridges.append(bridge)

    starts = {}
    for b in bridges:
        starts.setdefault((b.chunk, b.local_identity), []).append(b)
    for key, group in starts.items():
        if len(group) > 1:
            raise ValueError(f"local_identity {key[1]} in chunk {key[0]} is bridged {len(group)} times: {group}")
    return bridges


class _ChainTracker:
    """One chain per fly, holding the local identity that fly currently has.

    The set of active chains is explicit state. It is never rebuilt from the rows
    produced so far, so a chunk that matched badly cannot silently empty the next one.
    """

    def __init__(self, first_chunk, local_identities, log):
        self.log = log
        self.lid_of = {}       # chain id -> local identity in the current chunk
        self.parked = []       # (Bridge, chain id) waiting for the bridge to end
        self.dropped = {}      # chain id -> (chunk, reason)
        self._counter = 0
        for lid in sorted(local_identities):
            self.start(first_chunk, lid)

    def start(self, chunk, lid):
        chain = f"f{self._counter}"
        self._counter += 1
        self.lid_of[chain] = int(lid)
        self.log.write(f"{chunk} - chain {chain} starts as local_identity {lid}\n")
        return chain

    def chain_of(self, lid):
        for chain, value in self.lid_of.items():
            if value == int(lid):
                return chain
        return None

    def active_lids(self):
        return set(self.lid_of.values())

    def park(self, bridge):
        chain = self.chain_of(bridge.local_identity)
        if chain is None:
            logger.warning("Bridge %s starts from a local identity that is not active", bridge)
            chain = self.start(bridge.chunk, bridge.local_identity)
        del self.lid_of[chain]
        self.parked.append((bridge, chain))
        self.log.write(
            f"{bridge.chunk} - chain {chain} (local_identity {bridge.local_identity}) parked until "
            f"chunk {bridge.chunk_after}, where it becomes {bridge.local_identity_after}\n"
        )

    def unpark(self, chunk):
        still_parked = []
        for bridge, chain in self.parked:
            if bridge.chunk_after == chunk:
                self.lid_of[chain] = bridge.local_identity_after
                self.log.write(
                    f"{chunk} - chain {chain} returns from bridge as local_identity "
                    f"{bridge.local_identity_after}\n"
                )
            else:
                still_parked.append((bridge, chain))
        self.parked = still_parked

    def advance(self, chain, lid_after):
        self.lid_of[chain] = int(lid_after)

    def drop(self, chain, chunk, reason):
        self.lid_of.pop(chain, None)
        self.dropped[chain] = (chunk, reason)
        logger.warning("chunk %s: chain %s dropped (%s)", chunk, chain, reason)
        self.log.write(f"{chunk} - chain {chain} dropped: {reason}\n")

    def reconcile(self, chunk, lids_present, inside_bridge):
        """Compare the active chains with the identities actually present in `chunk`.

        Outside a bridge every identity should belong to a chain. Inside a bridge the
        bridged flies still have tracks, which are expected to belong to no chain.
        """
        active = self.active_lids()
        unexpected = sorted(set(lids_present) - active)
        vanished = sorted(active - set(lids_present))

        for lid in vanished:
            self.drop(self.chain_of(lid), chunk, f"local_identity {lid} is not present in chunk {chunk}")

        if unexpected and not inside_bridge:
            # not a bridge, so these are real flies nobody is following: adopt them
            logger.warning("chunk %s: local identities %s belong to no chain; starting new chains",
                           chunk, unexpected)
            for lid in unexpected:
                self.start(chunk, lid)
        elif unexpected:
            self.log.write(
                f"{chunk} - local identities {unexpected} belong to no chain (bridged flies' tracks), ignored\n"
            )

        if not self.lid_of:
            raise Exception(
                f"chunk {chunk}: no active chains left. Present here: {sorted(lids_present)}, "
                f"parked: {[(b, c) for b, c in self.parked]}, dropped: {self.dropped}"
            )
        return unexpected


def _match_engaged_pairs(chunk, next_chunk, tracker, engaged_labels, reserved, used, log):
    """Pair engaged-engaged local identities per bout: they sit at the bbox centroid,
    so distance carries no information."""
    rows, handled = [], set()
    if not engaged_labels:
        return rows, handled

    for interval_id, info in engaged_labels.items():
        engaged_now = info["engaged_per_chunk"].get(chunk)
        engaged_next = info["engaged_per_chunk"].get(next_chunk)
        if not (engaged_now and engaged_next):
            continue
        active = tracker.active_lids()
        sources = sorted(int(l) for l in engaged_now if int(l) in active and int(l) not in handled)
        targets = sorted(int(l) for l in engaged_next if int(l) not in used and int(l) not in reserved)
        if len(sources) != len(targets):
            logger.warning("chunk %s interval %s: %d engaged sources but %d free targets",
                           chunk, interval_id, len(sources), len(targets))
        for src, tgt in zip(sources, targets):
            rows.append((chunk, src, tgt, 0.0))
            used.add(tgt)
            handled.add(src)
            log.write(f"{chunk} - {src} -> {next_chunk} - {tgt} (engaged pair, interval_id={interval_id})\n")
    return rows, handled


def _merge_annotations(identity_table, annotated_table, excused):
    """Annotations win over automatic links that leave from, or arrive at, the same fly."""
    annotated_table = annotated_table.copy()
    annotated_table["distance"] = np.nan
    annotated_table["is_inferred"] = False
    annotated_table["priority"] = np.inf

    merged = pd.concat([annotated_table, identity_table], axis=0).sort_values(
        ["chunk", "priority", "local_identity"], ascending=[True, False, True]
    )
    for keys in (["chunk", "local_identity"], ["chunk_after", "local_identity_after"]):
        dups = merged.duplicated(keys)
        if dups.any():
            for row in merged.loc[dups].itertuples(index=False):
                excused[(int(row.chunk), int(row.local_identity))] = (
                    f"dropped after merge: {keys} collides with an annotated link "
                    f"({int(row.chunk)}:{int(row.local_identity)} -> "
                    f"{int(row.chunk_after)}:{int(row.local_identity_after)})"
                )
            logger.warning("Dropping %d rows with duplicated %s:\n%s",
                           int(dups.sum()), keys, merged.loc[dups].to_string())
            merged = merged.loc[~dups]

    return merged.sort_values(["chunk", "local_identity"])


def _check_no_fly_lost(identity_table, sources_seen, excused, bridges):
    """Every (chunk, local_identity) seen as a source must end up with a link in the
    final table, or have a recorded reason for not having one."""
    final = set(zip(identity_table["chunk"].astype(int), identity_table["local_identity"].astype(int)))
    bridged = {(b.chunk, b.local_identity) for b in bridges}

    rows = []
    for chunk in sorted(sources_seen):
        for lid in sorted(sources_seen[chunk]):
            if (chunk, lid) in final or (chunk, lid) in bridged:
                continue
            reason = excused.get((chunk, lid), "NO REASON RECORDED")
            expected = reason.startswith(("bridged", "track of a bridged fly"))
            rows.append(dict(chunk=chunk, local_identity=lid, reason=reason,
                             unexpected="" if expected else "<-- unexpected"))
    if not rows:
        return

    lost = pd.DataFrame(rows)
    print("=" * 100)
    print(f"{len(lost)} fly/chunk source(s) have no link in the final table:")
    with pd.option_context("display.max_rows", None, "display.width", None, "display.max_colwidth", None):
        print(lost.to_string(index=False))
    unexpected = lost[lost["unexpected"] != ""]
    if len(unexpected):
        print(f"\n{len(unexpected)} of these are not explained by a bridge and will break the chain "
              f"from chunk {int(unexpected['chunk'].min())} onwards.")
    print("=" * 100)


def match_animals_between_chunks_by_distance(before, after, local_identity_before, chunk,
                                             log=None, candidate_lids=None):
    """
    Match `local_identity_before` to whichever local_identity in `after`
    minimizes Euclidean distance.

    `candidate_lids` (optional): restrict the search to this list. Useful for
    excluding engaged flies' centroid positions (which collapse all distances
    to zero) and for excluding targets already paired earlier in the same
    chunk transition.
    """
    animal = before.loc[before["local_identity"] == local_identity_before]
    min_distance = math.inf
    selected_lid = None

    if candidate_lids is None:
        lids_series = after["local_identity"]
        if isinstance(lids_series, cudf.Series):
            lids_series = lids_series.to_pandas()
        lids = lids_series.unique().tolist()
    else:
        lids = list(candidate_lids)

    for lid_after in lids:
        next_animal = after.loc[after["local_identity"] == lid_after]
        if next_animal.shape[0] > 1:
            raise ValueError(
                f"{next_animal.shape[0]} animals found with local identity "
                f"{lid_after} in chunk {chunk+1}"
            )
        elif next_animal.shape[0] == 0:
            raise ValueError(
                f"0 animals found with local identity {lid_after} in chunk {chunk+1}"
            )

        distance = euclidean_distance(
            animal[["x", "y"]].values,
            next_animal[["x", "y"]].values,
        ).item()
        if distance < min_distance:
            min_distance = distance
            selected_lid = lid_after

    if log is not None:
        log.write(f"{chunk} - {local_identity_before} -> {chunk+1} - {selected_lid}\n")

    return selected_lid, min_distance



def make_identity_table(lid_table, annotated_table, chunks,
                        all_intervals_engaged_labels=None,
                        verbose=False, debug=False):
    """
    Build the chunk-to-chunk identity chain.

    One chain is tracked per fly. Annotated links spanning more than one boundary
    (bridges, e.g. copulation) park their chain at the start chunk and restore it at
    the end chunk, so the flies that are not copulating keep being matched normally in
    the chunks in between. A fly is never dropped without a recorded reason.

    lid_table: two rows per fly and chunk (position=first|last) with x, y, local_identity.
    Returns one row per fly per chunk transition, with local_identity_after.
    """
    chunks = [int(c) for c in chunks]
    chunk_set = set(chunks)
    missing = sorted(set(range(chunks[0], chunks[-1] + 1)) - chunk_set)
    if missing:
        logger.warning("These chunks are absent from `chunks`; no link will leave from them: %s", missing)

    annotated_table = (
        None if annotated_table is None else
        annotated_table.loc[
            (annotated_table["chunk"] >= chunks[0]) & (annotated_table["chunk_after"] <= chunks[-1])
        ].copy()
    )
    bridges = _parse_bridges(annotated_table, chunks)

    records, sources_seen, excused = [], {}, {}

    with open("identity_table.log", "w") as log:
        _, first_lids = _local_identities(lid_table, chunks[0], "last")
        tracker = _ChainTracker(chunks[0], first_lids, log)

        for chunk in tqdm(chunks[:-1]):
            next_chunk = chunk + 1
            if next_chunk not in chunk_set:
                logger.warning("chunk %s is absent from `chunks`; links from chunk %s may dead-end there",
                               next_chunk, chunk)

            tracker.unpark(chunk)

            before, lids_before = _local_identities(lid_table, chunk, "last")
            after, lids_after = _local_identities(lid_table, next_chunk, "first")
            sources_seen.setdefault(chunk, set()).update(lids_before)

            inside_bridge = any(b.chunk < chunk < b.chunk_after for b in bridges)
            ignored = tracker.reconcile(chunk, lids_before, inside_bridge)
            for lid in ignored:
                if inside_bridge:
                    excused[(chunk, lid)] = "track of a bridged fly, not followed by any chain"

            # identities reserved in next_chunk by a bridge landing there
            reserved = {b.local_identity_after for b in bridges if b.chunk_after == next_chunk}
            used = set(reserved)

            # bridges leaving from this chunk: park their chains, they are already linked
            for bridge in [b for b in bridges if b.chunk == chunk]:
                tracker.park(bridge)
                excused[(chunk, bridge.local_identity)] = f"bridged by annotation to chunk {bridge.chunk_after}"

            # --- Phase 1: engaged-engaged pairs across the boundary, per bout
            engaged_rows, handled = _match_engaged_pairs(
                chunk, next_chunk, tracker, all_intervals_engaged_labels, reserved, used, log
            )
            for row in engaged_rows:
                chain = tracker.chain_of(row[1])
                if chain is not None:
                    tracker.advance(chain, row[2])
            records.extend(engaged_rows)

            # --- Phase 2: distance-match the remaining chains
            for chain, local_identity in sorted(tracker.lid_of.items(), key=lambda kv: kv[1]):
                if local_identity in handled:
                    continue

                candidate_lids = [lid for lid in lids_after if lid not in used]
                if not candidate_lids:
                    reason = (f"no free identity left in chunk {next_chunk}; "
                              f"{sorted(lids_after)} all taken (reserved by a bridge: {sorted(reserved)})")
                    excused[(chunk, local_identity)] = reason
                    tracker.drop(chain, chunk, reason)
                    continue

                local_identity_after, min_distance = match_animals_between_chunks_by_distance(
                    before, after, local_identity, chunk, log, candidate_lids=candidate_lids,
                )
                local_identity_after = int(local_identity_after)

                if local_identity_after in used:
                    # candidates exclude `used`, so this means the matcher ignored them
                    logger.warning("%s already used in chunk %s", local_identity_after, chunk)
                    log.write(f"{local_identity_after} already used in chunk {chunk}\n")
                    if verbose:
                        print(before, after)
                    if debug:
                        import ipdb; ipdb.set_trace()
                else:
                    used.add(local_identity_after)

                if verbose:
                    print(f"chunk {chunk} - {local_identity} > {local_identity_after}")
                records.append((chunk, local_identity, local_identity_after, min_distance))
                tracker.advance(chain, local_identity_after)

    identity_table = pd.DataFrame.from_records(
        records, columns=["chunk", "local_identity", "local_identity_after", "distance"]
    )
    identity_table["chunk"] = identity_table["chunk"].astype(int)
    identity_table["is_inferred"] = False
    identity_table["chunk_after"] = identity_table["chunk"] + 1
    identity_table["priority"] = 2
    identity_table.to_csv("identity_table_before_annotated_table.csv")

    if annotated_table is not None:
        identity_table = _merge_annotations(identity_table, annotated_table, excused)

    identity_table.to_csv("identity_table.csv")
    _check_no_fly_lost(identity_table, sources_seen, excused, bridges)
    ensure_continuity_of_table(identity_table)
    return identity_table