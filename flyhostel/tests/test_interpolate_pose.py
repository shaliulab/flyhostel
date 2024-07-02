import unittest
import pandas as pd
from flyhostel.data.pose.filters import interpolate_pose  # Import the function from your code module

class TestInterpolatePose(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame to use in tests
        self.pose = pd.DataFrame({
            'A_x': [1, None, 3, None, None, None],
            'A_y': [1, None, 3, None, None, None],
            'B_x': [4, 5, None, 1, 2, None],
            'B_y': [4, 5, None, 1, 2, None],
            'C_x': [None, 7, 9, None, None, None],
            'C_y': [None, 7, 9, None, None, None]
        })
        self.columns = ['A', 'B']

    # def test_default_parameters(self):
    #     result = interpolate_pose(self.pose.copy(), pose_framerate=1)
    #     # Check if all NaN values are interpolated
    #     self.assertFalse(result.isnull().values.any())

    # def test_specific_columns(self):
    #     result = interpolate_pose(self.pose.copy(), columns=self.columns, pose_framerate=1)
    #     # Check if specified columns are interpolated
    #     self.assertFalse(result[self.columns].isnull().values.any())
    #     # Check if unspecified column is not interpolated
    #     self.assertTrue(result['C'].isnull().values.any())

    # def test_float_seconds(self):
    #     result = interpolate_pose(self.pose.copy(), seconds=1, pose_framerate=1)
    #     # Add assertions based on expected behavior with seconds as float

    # def test_none_seconds(self):
    #     result = interpolate_pose(self.pose.copy(), seconds=None, pose_framerate=1)
    #     # Add assertions for behavior with seconds as None

    def test_dict_seconds(self):
        seconds_dict = {'A': 3, 'C': 2}
        result = interpolate_pose(self.pose.copy(), seconds=seconds_dict, pose_framerate=1)
        print(result)

        # Add assertions for behavior with seconds as dict

    # def test_invalid_input(self):
    #     with self.assertRaises(ValueError):
    #         interpolate_pose(self.pose.copy(), columns=['X'], pose_framerate=1)

    # def test_inplace_modification(self):
    #     original_pose = self.pose.copy()
    #     interpolate_pose(original_pose, pose_framerate=1)
    #     # Check if original_pose is modified
    #     self.assertNotEqual(original_pose, self.pose)

if __name__ == '__main__':
    unittest.main()
