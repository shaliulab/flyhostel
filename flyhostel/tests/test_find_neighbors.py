import unittest
import cudf
from flyhostel.data.interactions.neighbors_gpu import find_neighbors

class TestInterpolatePose(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame to use in tests
        self.pose = cudf.DataFrame({
            'centroid_x': [1, 1, 1, 3, 1.5, 3],
            'centroid_y': [0, 0, 0, 0, 0, 0],
            'id': ["A", "A", "A", "B", "B", "B"],
            'frame_number': [1, 2, 3, 1, 2, 3]
        })


    def test_find_neighbors(self):
        result = find_neighbors(self.pose, 1)
        self.assertEqual(result["distance"].iloc[0], 0.5)
        self.assertEqual(result["distance"].iloc[1], 0.5)
        self.assertEqual(result["id"].iloc[0], "A")
        self.assertEqual(result["id"].iloc[1], "B")

if __name__ == '__main__':
    unittest.main()
