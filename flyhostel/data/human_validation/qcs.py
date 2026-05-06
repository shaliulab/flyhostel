
# 0 local_identity
# 1 identity
# 2 chunk
# 3 fragment
# 4 modified
# 5 frame_number
# 6 x
# 7 y
import numpy as np

INDICES={
    "identity": 0,
    "chunk": 2,
    "fragment": 3,
    "modified": 4,
    "frame_number":5,
}

def yolov7_qc(window):
    """
    Return True if YOLOv7 did not process any frame in the group
    """
    return (window[:, INDICES["modified"]]==0).all()


def all_found_qc(window, number_of_animals):
    """
    Return True if the number of found objects in all frames of the group
    matches the number of animals
    """
    return window.shape[0]==number_of_animals


def all_id_expected_qc(window, number_of_animals, idx=INDICES["identity"]):
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
    return window[0, INDICES["frame_number"]] % chunksize != 0

def last_frame_idx_qc(window, chunksize):
    """
    Return True if this window is not in the first frame of the chunk
    """
    return window[0, INDICES["frame_number"]] % chunksize != (chunksize-1)


def inter_qc(window_before, window):
    """
    Return False if the fragment identifiers in the two windows are not the same
    In other words, flag a fragment change
    In other words, return True if all fragment identifiers are the same 
    """

    fragment_identifiers_constant = set(window[:, INDICES["fragment"]]).issubset(window_before[:, INDICES["fragment"]])
    return fragment_identifiers_constant


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

