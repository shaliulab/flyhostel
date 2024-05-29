import numpy as np
from tqdm.auto import tqdm
import joblib

# Extract edges for each individual
def get_edges_from_row(row, bodyparts):
    points = []
    ref_bodypart="thorax"
    ref_point=(row[f"{ref_bodypart}_x"], row[f"{ref_bodypart}_y"])

    for bodypart in bodyparts:
        x = row[f'{bodypart}_x']
        y = row[f'{bodypart}_y']
        points.append((x, y))
        
    edges = [(ref_point, points[i]) for i in range(len(points))]
    return edges

def preprocess_data(df, bodyparts):

    df['edges'] = df.apply(lambda x: get_edges_from_row(x, bodyparts=bodyparts), axis=1)

    return df



def point_distance(A, B):
    return np.linalg.norm(np.array(A) - np.array(B))

def point_to_segment_distance(P, A, B):
    # Vector AB
    AB = np.array(B) - np.array(A)
    # Vector AP
    AP = np.array(P) - np.array(A)
    # Project AP onto AB, normalized by the length of AB
    t = np.dot(AP, AB) / np.dot(AB, AB)
    t = np.clip(t, 0, 1)
    # Closest point on segment AB to point P
    closest = A + t * AB
    return point_distance(P, closest)

def segments_intersect_or_distance(A, B, C, D):
    if ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D):
        return True, 0.0  # Segments intersect
    else:
        # Segments do not intersect; calculate the minimum distance between them
        distances = [
            point_to_segment_distance(A, C, D),
            point_to_segment_distance(B, C, D),
            point_to_segment_distance(C, A, B),
            point_to_segment_distance(D, A, B)
        ]
        min_distance = min(distances)
        return False, min_distance


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def segments_intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def are_touching(edges1, edges2):
    min_distance=np.inf
    for edge1 in edges1:
        for edge2 in edges2:
            intersect, distance = segments_intersect_or_distance(edge1[0], edge1[1], edge2[0], edge2[1])
            if intersect:
                return 0
            else:
                min_distance=min(min_distance, distance)

    return min_distance

def check_intersection(frame, frame_data):
    ids = frame_data['id'].values
    pair_ids=frame_data["nn"].values
    edges_dict = {row['id']: row['edges'] for _, row in frame_data.iterrows()}
    results=[]
    for id1, id2 in zip(ids, pair_ids):
        distance=are_touching(edges_dict[id1], edges_dict[id2])
        results.append((frame, id1, id2, distance))

    return results


# def check_intersection(frame, frame_data, mask):
#     frame_data=frame_data.merge(mask[["id", "nn", "frame_number"]], on=["id", "frame_number"], how="inner")
#     ids = frame_data['id'].values
#     pair_ids=frame_data["nn"].values
#     # if frame == 5095046:
#     #     import ipdb; ipdb.set_trace()

#     pairs=zip(ids, pair_ids)

#     edges_dict = {row['id']: row['edges'] for _, row in frame_data.iterrows()}
#     results=[]

#     for id1, id2 in pairs:
#         distance=are_touching(edges_dict[id1], edges_dict[id2])
#         results.append((frame, id1, id2, distance))

#     return results

def check_intersections(df, mask, n_jobs=1):
    frames = df['frame_number'].unique()
    
    results=joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(
            check_intersection
        )(
            frame, df[df['frame_number'] == frame], #mask[mask['frame_number'] == frame]
        )
        for frame in tqdm(frames, desc="Asserting touch")
    )

    return results