import logging
import numpy as np
import pandas as pd
logger=logging.getLogger(__name__)

LINK_COLUMNS = ["chunk", "local_identity", "chunk_after", "local_identity_after"]


def _as_int(x):
    return None if pd.isna(x) else int(x)


def _fmt(node):
    return f"{node[0]}:{node[1]}"


def _build_links(table):
    """
    forward[(chunk, lid)]  -> list of (chunk_after, lid_after)
    backward[(chunk, lid)] -> list of (chunk_before, lid_before)
    Every (chunk, local_identity) row of the table is a key of `forward`,
    even when it has no outgoing link (empty list).
    """
    forward, backward = {}, {}
    for chunk, lid, chunk_after, lid_after in table[LINK_COLUMNS].itertuples(index=False, name=None):
        src = (int(chunk), int(lid))
        targets = forward.setdefault(src, [])
        chunk_after, lid_after = _as_int(chunk_after), _as_int(lid_after)
        if chunk_after is None or lid_after is None:
            continue
        dst = (chunk_after, lid_after)
        targets.append(dst)
        backward.setdefault(dst, []).append(src)
    return forward, backward


def _trace_chain(start, forward):
    """Same walk as the original while-loop, but it records why it stopped
    and does not loop forever on a link that doesn't move forward in time."""
    steps = [start]       # (chunk, lid) nodes actually visited
    chunks = [start[0]]   # as in the original: chunks skipped by a link are filled in
    node = start
    while True:
        targets = forward.get(node, [])
        if len(targets) == 0:
            reason = f"no link out of {_fmt(node)}"
            if node not in forward:
                reason += " (this node is not a row of the table)"
            return steps, np.array(chunks), reason
        if len(targets) > 1:
            return steps, np.array(chunks), f"ambiguous: {_fmt(node)} -> {', '.join(map(_fmt, targets))}"
        nxt = targets[0]
        if nxt[0] <= node[0]:
            return steps, np.array(chunks), f"link does not move forward in time: {_fmt(node)} -> {_fmt(nxt)}"
        chunks.extend(range(chunks[-1] + 1, nxt[0] + 1))
        steps.append(nxt)
        node = nxt


def _link_origins(table):
    origin = {}
    has_p, has_d = "priority" in table.columns, "distance" in table.columns
    for row in table.itertuples(index=False):
        if pd.isna(row.chunk_after) or pd.isna(row.local_identity_after):
            continue
        key = ((int(row.chunk), int(row.local_identity)), (int(row.chunk_after), int(row.local_identity_after)))
        if has_p and np.isinf(row.priority):
            origin[key] = "annotated"
        elif has_d and row.distance == 0:
            origin[key] = "engaged pair"
        elif has_d and not pd.isna(row.distance):
            origin[key] = f"distance-matched, d={row.distance:.1f}"
        else:
            origin[key] = "unknown origin"
    return origin


def _table_problems(table, forward, backward):
    origin = _link_origins(table)
    link = lambda s, d: f"{_fmt(s)} -> {_fmt(d)} ({origin.get((s, d), '?')})"
    lids_by_chunk = {int(c): sorted(set(map(int, s))) for c, s in table.groupby("chunk")["local_identity"]}
    last_chunk = max(lids_by_chunk)
    problems = []

    for src, targets in sorted(forward.items()):
        if len(targets) > 1:
            problems.append(f"{_fmt(src)} has {len(targets)} exits, a chain can only follow one: "
                            + "; ".join(link(src, d) for d in targets))
        for dst in targets:
            if dst[0] <= src[0]:
                problems.append(f"{link(src, dst)} goes backwards in time")
            elif dst[0] != src[0] + 1 and origin.get((src, dst)) != "annotated":
                problems.append(f"{link(src, dst)} skips chunks but is not an annotation")

    for dst, sources in sorted(backward.items()):
        if len(sources) > 1:
            problems.append(f"{_fmt(dst)} is reached by {len(sources)} flies: "
                            + "; ".join(link(s, dst) for s in sources))

    stuck = {}
    for src, targets in forward.items():
        for dst in targets:
            if dst not in forward and dst[0] <= last_chunk:
                stuck.setdefault(dst[0], []).append((src, dst))
    for chunk, items in sorted(stuck.items()):
        arrivals = "; ".join(link(s, d) for s, d in sorted(items))
        if chunk not in lids_by_chunk:
            problems.append(f"chunk {chunk} has no rows at all, so no fly can continue past it "
                            f"(was it in `chunks`?). Flies arriving there: {arrivals}")
        else:
            stuck_lids = sorted({d[1] for _, d in items})
            problems.append(f"chunk {chunk} has links out of local_identity {lids_by_chunk[chunk]} only, "
                            f"so {stuck_lids} are dead ends. Flies arriving there: {arrivals}")

    n_max = max(len(v) for v in lids_by_chunk.values())
    fewer = {c: v for c, v in lids_by_chunk.items() if len(v) < n_max}
    if fewer:
        problems.append(f"chunks with fewer than {n_max} flies linked forward: "
                        + ", ".join(f"{c} (only {v})" for c, v in fewer.items()))
    return problems


def _build_detail(chains, forward, backward, owners):
    rows = []
    for label, chain in chains.items():
        visited = dict(chain["steps"])  # chunk -> lid
        for chunk in map(int, chain["chunks"]):
            if chunk not in visited:
                rows.append(dict(fly=label, chunk=chunk, local_identity=pd.NA,
                                 previous="", next="", flags="chunk skipped by a link"))
                continue
            node = (chunk, visited[chunk])
            prev, nxt = backward.get(node, []), forward.get(node, [])
            flags = []
            if len(prev) > 1:
                flags.append("several predecessors")
            if len(nxt) > 1:
                flags.append("several successors")
            others = [o for o in owners[node] if o != label]
            if others:
                flags.append("also claimed by " + ",".join(others))
            rows.append(dict(
                fly=label, chunk=chunk, local_identity=node[1],
                previous=", ".join(map(_fmt, prev)) or "-",
                next=", ".join(map(_fmt, nxt)) or "-",
                flags="; ".join(flags),
            ))
    return pd.DataFrame(rows)


#######################################

def _build_grid(table, chains, forward, backward, owners):
    all_chunks = set(table["chunk"].astype(int))
    for chain in chains.values():
        all_chunks.update(int(c) for c in chain["chunks"])
    for targets in forward.values():
        all_chunks.update(c for c, _ in targets)
    index = sorted(all_chunks)
    last_chunk = max(index)

    grid = pd.DataFrame(".", index=pd.Index(index, name="chunk"), columns=list(chains))
    notes = {c: [] for c in index}

    # one number per (chunk, fly)
    for label, chain in chains.items():
        steps = chain["steps"]
        for chunk, lid in steps:
            grid.at[chunk, label] = str(lid)
        for a, b in zip(steps, steps[1:]):
            if b[0] != a[0] + 1:
                notes[a[0]].append(f"{label} jumps {_fmt(a)} -> {_fmt(b)}, skipping chunks in between")
        end = steps[-1]
        if end[0] != last_chunk or forward.get(end):
            notes[end[0]].append(f"{label} stops: {chain['stop_reason']}")

    # anomalies in the links
    for node in sorted(forward):
        chunk, lid = node
        prev, nxt = backward.get(node, []), forward[node]
        if len(prev) > 1:
            notes[chunk].append(f"merge: {', '.join(map(_fmt, prev))} both -> {_fmt(node)}")
        if len(nxt) > 1:
            notes[chunk].append(f"fork: {_fmt(node)} -> {', '.join(map(_fmt, nxt))}")
        if node not in owners:
            notes[chunk].append(f"lid {lid} not reached by any chain")
        elif len(owners[node]) > 1 and not any(set(owners.get(p, [])) == set(owners[node]) for p in prev):
            notes[chunk].append(f"{', '.join(owners[node])} follow the same animal from here")
        for dst in nxt:
            if dst not in forward and dst[0] <= last_chunk:
                notes[chunk].append(f"{_fmt(node)} -> {_fmt(dst)}: target is not a row of the table")

    grid["notes"] = ["; ".join(notes[c]) for c in index]
    return grid


def _print_report(table, chains, problems, grid, detail, failed, number_of_chunks):
    bar = "=" * 100
    with pd.option_context("display.max_rows", None, "display.max_columns", None,
                           "display.width", None, "display.max_colwidth", None):
        print(bar)
        print("ensure_continuity_of_table: " + ("FAILED. Is validation_lags.csv correct?" if failed else "OK"))
        print(bar)
        print(f"{number_of_chunks} chunks in table ({table['chunk'].min()}..{table['chunk'].max()}), "
              f"{len(chains)} chains (one per local_identity, starting at its first chunk)")

        print("\nChains:")
        for label, chain in chains.items():
            status = "FAIL" if chain["issues"] else "ok"
            print(f"  {label} [{status}] starts as local_identity {chain['steps'][0][1]} "
                  f"in chunk {chain['steps'][0][0]}, ends at {_fmt(chain['steps'][-1])}")
            for issue in chain["issues"]:
                print(f"      - {issue}")

        if problems:
            print(f"\nProblems in the link table ({len(problems)}):")
            for p in problems:
                print("  - " + p)

        print("\nReconstructed identity grid")
        print("  each cell = local_identity of that fly in that chunk")
        print("  read a column top to bottom to follow one fly across chunks")
        print("  '.' = the fly's chain does not reach this chunk | '!' = see notes below")
        shown = grid.drop(columns="notes")
        shown["!"] = np.where(grid["notes"] != "", "!", "")
        print(shown.to_string())

        noted = grid[grid["notes"] != ""]
        if len(noted):
            print("\nNotes by chunk:")
            for chunk, text in noted["notes"].items():
                for note in text.split("; "):
                    print(f"  chunk {chunk}: {note}")

        if failed:
            print(f"\nDetail of failing chains ({', '.join(failed)}):")
            print(detail[detail["fly"].isin(failed)].to_string(index=False))
        print(bar)

#######################################

def ensure_continuity_of_table(table, verbose=False, report_path=None):
    """
    verbose:      print the full report even when everything is fine
    report_path:  if given, write <report_path>_grid.csv and <report_path>_detail.csv
    Returns {local_identity: array of chunks}, like `all_chunks` in the original.
    """
    local_identities = table["local_identity"].drop_duplicates().tolist()
    table.sort_values(["chunk", "local_identity"], inplace=True)
    number_of_chunks = table["chunk"].nunique()

    forward, backward = _build_links(table)

    chains = {}
    for i, lid in enumerate(local_identities):
        first_chunk = int(table.loc[table["local_identity"] == lid, "chunk"].iloc[0])
        steps, chunks, stop_reason = _trace_chain((first_chunk, int(lid)), forward)
        chains[f"F{i}"] = dict(lid=lid, steps=steps, chunks=chunks, stop_reason=stop_reason, issues=[])

    owners = {}
    for label, chain in chains.items():
        for node in chain["steps"]:
            owners.setdefault(node, []).append(label)

    for label, chain in chains.items():
        n_transitions = len(chain["chunks"]) - 1  # == len(np.diff(chunks)) in the original check
        if n_transitions != number_of_chunks:
            chain["issues"].append(
                f"{n_transitions} chunk transitions, expected {number_of_chunks} (stopped: {chain['stop_reason']})"
            )
        shared = [n for n in chain["steps"] if len(owners[n]) > 1]
        if shared:
            others = sorted({o for n in shared for o in owners[n]} - {label})
            chain["issues"].append(
                f"shares {len(shared)} node(s) with {','.join(others)}, first at {_fmt(shared[0])}"
            )

    failed = [label for label, chain in chains.items() if chain["issues"]]

    if failed or verbose or report_path:
        problems = _table_problems(table, forward, backward)
        grid = _build_grid(table, chains, forward, backward, owners)
        detail = _build_detail(chains, forward, backward, owners)
        if failed or verbose:
            _print_report(table, chains, problems, grid, detail, failed, number_of_chunks)
        if report_path:
            grid.to_csv(f"{report_path}_grid.csv")
            detail.to_csv(f"{report_path}_detail.csv", index=False)

    if failed:
        summary = "; ".join(
            f"{label} (local_identity {chains[label]['lid']}, start {_fmt(chains[label]['steps'][0])}): "
            + " | ".join(chains[label]["issues"])
            for label in failed
        )

        msg=f"Identity chaining failed for {len(failed)}/{len(chains)} chains: {summary}"
        logger.error(msg)

    return {chain["lid"]: chain["chunks"] for chain in chains.values()}