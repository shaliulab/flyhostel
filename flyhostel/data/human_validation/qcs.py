
# 0 local_identity
# 1 identity
# 2 chunk
# 3 fragment
# 4 modified
# 5 frame_number
# 6 x
# 7 y

identity_idx=0
chunk_idx=2
fragment_idx=3
modified_idx=4
frame_number_idx=5

def yolov7_qc(window):
    """
    Return True if YOLOv7 did not process any frame in the group
    """
    return (window[:, modified_idx]==0).all()


def all_found_qc(window, number_of_animals):
    """
    Return True if the number of found objects in all frames of the group
    matches the number of animals
    """
    return window.shape[0]==number_of_animals


def all_id_expected_qc(window, number_of_animals, idx=identity_idx):
    """
    Return True only if the local identities found are the ones expected
    from the number of animals
    So if there are three animals, the local identities available shoould be 1 2 and 3
    """
    if number_of_animals > 1:
        identities=range(1, number_of_animals+1)
    else:
        identities=[0]
    expected=[i for i in identities]
    return (sorted(window[:, idx])==expected)


def first_frame_idx_qc(window, chunksize):
    """
    Return True if this window is not in the first frame of the chunk
    """
    return window[0, frame_number_idx] % chunksize != 0

def last_frame_idx_qc(window, chunksize):
    """
    Return True if this window is not in the first frame of the chunk
    """
    return window[0, frame_number_idx] % chunksize != (chunksize-1)


def inter_qc(window_before, window, window_after):
    """
    Return False if the fragment identifiers in the two windows are not the same
    In other words, flag a fragment change
    In other words, return True if all fragment identifiers are the same 
    """

    is_different = not set(window[:, fragment_idx]).issubset(window_before[:, fragment_idx])
    if window_after is None:
        is_end_of_chunk=False
    else:
        is_end_of_chunk = window[0, chunk_idx] < window_after[0, chunk_idx]

    return not is_different and not is_end_of_chunk


def intra_qc(window, number_of_animals):
    """
    Return True if all of these conditions are met, and False otherwise
        All local identities are non zero
        modified is set to 0 (meaning YOLOv7 has not processed the frame)
        The number of detected animals matches the expected number of animals
    """
    return yolov7_qc(window) and all_found_qc(window, number_of_animals) and all_id_expected_qc(window, number_of_animals) and nms_qc(window)

def nms_qc(window):
    """
    Return True if two objects have the same centroid (non maximal suppresion failure)
    """

    x_y=window[:, 6:7]
    u, c = np.unique(x_y, return_counts=True, axis=0)
    duplicates=np.sum(c > 1)
    return not duplicates


