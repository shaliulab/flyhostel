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


if __name__ == "__main__":
    pytest.main()
