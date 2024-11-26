import cudf
import cupy as cp
import numpy as np
import pandas as pd
import itertools
import pytest
from unittest.mock import patch, MagicMock
from flyhostel.data.interactions.neighbors_gpu import compute_distance_between_all_ids, compute_distance_between_pairs

# Sample data and helper function for distance calculation
def create_sample_data(ids, frame_numbers, centroids):
    """Creates a cudf DataFrame with columns: id, frame_number, centroid_x, centroid_y."""
    data = {
        "id": ids,
        "frame_number": frame_numbers,
        "centroid_x": [centroid[0] for centroid in centroids],
        "centroid_y": [centroid[1] for centroid in centroids]
    }
    print(data)
    return cudf.DataFrame(data)

# Helper function to create dummy distances for mock testing compute_distance_between_pairs
def mock_compute_distance_between_pairs(df, id1, id2, **kwargs):
    return cp.array([1.0, 2.0, 3.0])  # example fixed distance array for simplicity

@pytest.fixture
def sample_data():
    ids = [str(e) for e in [1, 2, 3, 1, 2, 3]]
    frame_numbers = [1, 1, 1, 2, 2, 2]
    centroids = [(10, 10), (20, 20), (30, 30), (15, 15), (25, 25), (35, 35)]
    return create_sample_data(ids, frame_numbers, centroids)

def test_compute_distance_between_all_ids_valid_input(sample_data):
    """Test that the function computes distance for valid input data without errors."""
    ids = ["1", "2", "3"]
    with patch("flyhostel.data.interactions.neighbors_gpu.compute_distance_between_pairs", side_effect=mock_compute_distance_between_pairs):
        distance_matrix = compute_distance_between_all_ids(sample_data, ids)
        assert distance_matrix.shape == (3, 2, 3)  # 3 ids, 2 neighbors, 3 frames/timestamps

def test_compute_distance_between_all_ids_no_pairs(sample_data):
    """Test that the function returns an empty distance matrix if there are no pairs."""
    ids = ["1"]  # Only one id, so no pairs
    with patch("flyhostel.data.interactions.neighbors_gpu.compute_distance_between_pairs", side_effect=mock_compute_distance_between_pairs):
        with pytest.raises(AssertionError) as excinfo:
            distance_matrix = compute_distance_between_all_ids(sample_data, ids)
        assert "Pass more than 1 id" in str(excinfo.value)

def test_compute_distance_between_all_ids_valid_input_real_data():
    real_data=cudf.DataFrame(
        pd.read_csv("data/interactions_real_data.csv", index_col=0)
    )
    frame_numbers=sorted(real_data["frame_number"].to_pandas().unique().tolist())
    ids=real_data["id"].to_pandas().unique().tolist()

    # assert real_data.groupby("frame_number").size().reset_index(drop=True).drop_duplicates().shape[0]==1
    
    distance_matrix = compute_distance_between_all_ids(real_data, ids=ids, step=10)
    n_frames=1
    assert cp.isinf(distance_matrix).sum()==len(ids)*(len(ids)-1)*n_frames
    count_missing=cp.isinf(distance_matrix).sum()
    assert count_missing==30
    print(count_missing)

    n0=500
    width=1000
    n1=n0+width
    real_data_subset=real_data.loc[real_data["frame_number"].isin(frame_numbers[n0:n1])]
    distance_matrix = compute_distance_between_all_ids(real_data_subset, ids)
    n_missing=cp.isinf(distance_matrix).sum()
    assert n_missing==0
    assert distance_matrix.shape == (len(ids), len(ids)-1, width)  # 3 ids, 2 neighbors

def test_compute_distance_between_all_ids_id_not_in_df(sample_data):
    """Test that the function correctly handles IDs that are not in the DataFrame."""
    ids = [1, 2, 4]  # 4 is not in the sample_data
    with patch("flyhostel.data.interactions.neighbors_gpu.compute_distance_between_pairs", side_effect=mock_compute_distance_between_pairs):
        distance_matrix = compute_distance_between_all_ids(sample_data, ids)
        assert distance_matrix.shape == (3, 2, 3)

# def test_compute_distance_between_all_ids_empty_df():
#     """Test that the function handles an empty DataFrame gracefully."""
#     df = cudf.DataFrame(columns=["id", "frame_number", "centroid_x", "centroid_y"])
#     ids = ["1", "2"]
#     with patch("flyhostel.data.interactions.neighbors_gpu.compute_distance_between_pairs", side_effect=mock_compute_distance_between_pairs):
#         distance_matrix = compute_distance_between_all_ids(df, ids)
#         assert distance_matrix.size == 0


# def test_compute_distance_between_all_ids_correct_values(sample_data):
#     """Test that the function returns expected values by mocking the distance calculation."""
#     ids = [1, 2, 3]
#     # Mock the compute_distance_between_pairs function to return a predictable result
#     with patch("flyhostel.data.interactions.neighbors_gpu.compute_distance_between_pairs", side_effect=mock_compute_distance_between_pairs) as mock_func:
#         distance_matrix = compute_distance_between_all_ids(sample_data, ids)
#         # Check that distances in the matrix are as expected from the mock
#         assert cp.allclose(distance_matrix, cp.array([[[1.0, 2.0, 3.0]] * 2] * 3))
#         mock_func.assert_called()  # Verify that our mock function was indeed called

def test_compute_distance_between_pairs():
    real_data=cudf.DataFrame(
        pd.read_csv("data/interactions_one_pair.csv", index_col=0)
    )
    dist=compute_distance_between_pairs(
        real_data,
        step=10,
    )
    print(cp.isinf(dist).sum())
    
if __name__ == "__main__":
    test_compute_distance_between_pairs()
    # pytest.main()
