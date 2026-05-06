from itertools import combinations


from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform

import numpy as np
from .utils import (
    _body_axis,
    _centroid,
)
from .data_structures import (
    FlyPose,
)



# ---------------------------------------------------------------------------
# Group-level features
# ---------------------------------------------------------------------------
def group_level_features(
        flies: list[FlyPose],
        contact_radius: float = 5.0,
) -> dict:
    n = len(flies)
    centroids = np.array([_centroid(f) for f in flies])   # (n, 2)
    headings  = np.array([_body_axis(f) for f in flies])  # (n, 2)

    # --- spatial spread ---
    group_centroid  = centroids.mean(axis=0)
    dists_to_center = np.linalg.norm(centroids - group_centroid, axis=1)
    spread_mean = float(dists_to_center.mean())
    spread_std  = float(dists_to_center.std())
    spread_max  = float(dists_to_center.max())

    # --- convex hull area ---
    hull_area = 0.0
    if n >= 3:
        try:
            hull = ConvexHull(centroids)
            hull_area = float(hull.volume)
        except Exception:
            hull_area = 0.0
    elif n == 2:
        hull_area = float(np.linalg.norm(centroids[0] - centroids[1]))

    # --- heading coherence (Vicsek order parameter) ---
    mean_heading_vec   = headings.mean(axis=0)
    heading_order      = float(np.linalg.norm(mean_heading_vec))
    polarization_angle = float(np.arctan2(mean_heading_vec[1],
                                          mean_heading_vec[0]))

    # --- pairwise distances ---
    pw_dists_flat = pdist(centroids)
    pw_dists      = squareform(pw_dists_flat)
    np.fill_diagonal(pw_dists, np.inf)
    nn_dists = pw_dists.min(axis=1)

    # --- contact graph ---
    contact_adj = (pw_dists < contact_radius).astype(int)
    np.fill_diagonal(contact_adj, 0)            # belt-and-suspenders

    degrees         = contact_adj.sum(axis=1)
    n_contact_edges = int(degrees.sum()) // 2
    max_edges       = n * (n - 1) / 2
    contact_density = n_contact_edges / max_edges if max_edges > 0 else 0.0
    degree_mean     = float(degrees.mean())
    degree_max      = int(degrees.max())
    degree_std      = float(degrees.std())

    # connectivity via BFS
    visited = set()
    stack   = [0]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(np.where(contact_adj[node] > 0)[0].tolist())
    is_connected = int(len(visited) == n)

    # --- compactness ---
    max_pw_dist = float(pw_dists_flat.max()) if n >= 2 else 1.0
    compactness = hull_area / (max_pw_dist ** 2 + 1e-9)

    return {
        "group_n":               n,
        "group_spread_mean":     spread_mean,
        "group_spread_std":      spread_std,
        "group_spread_max":      spread_max,
        "group_hull_area":       hull_area,
        "group_heading_order":   heading_order,
        "group_polar_angle":     polarization_angle,
        "group_nn_dist_mean":    float(nn_dists.mean()),
        "group_nn_dist_std":     float(nn_dists.std()),
        "group_n_contact_edges": n_contact_edges,
        "group_contact_density": contact_density,
        "group_degree_mean":     degree_mean,
        "group_degree_max":      degree_max,
        "group_degree_std":      degree_std,
        "group_is_connected":    is_connected,
        "group_compactness":     compactness,
    }
