import unittest
import cudf
from flyhostel.data.interactions.neighbors_gpu import find_neighbors, compute_pairwise_distances_using_bodyparts_gpu

class TestInterpolatePose(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame to use in tests
        self.dt = cudf.DataFrame({
            'centroid_x': [0, 1, 1, 1, 1, 1, 1, 0,  3, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 3],
            'centroid_y': [0, 0, 0, 0, 0, 0, 0, 0,  0,  0,   0,   0,   0,   0,  0,   0],
            'id': ["A", "A", "A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B", "B", "B"],
            'frame_number': [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]
        })

        self.pose = cudf.DataFrame({
            'thorax_x': [0, 1, 1, 1, 1, 1, 1, 0,  3, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 3],
            'thorax_y': [0, 0, 0, 0, 0, 0, 0, 0,  0,  0,   0,   0,   0,   0,  0,   0],
            'id': ["A", "A", "A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B", "B", "B"],
            'frame_number': [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]
        })


    def test_find_neighbors(self):
        neighbors = find_neighbors(self.dt, 1)
        result=compute_pairwise_distances_using_bodyparts_gpu(
            neighbors, self.pose,
            bodyparts=["thorax"],
            bodyparts_xy=["thorax_x", "thorax_y"]
        )
        print(result)

if __name__ == '__main__':
    unittest.main()
