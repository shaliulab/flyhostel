import unittest

import numpy as np
from flyhostel.quantification.sleep import make_waffle_array

NROWS = 10
NCOLS = 30


class TestMakeWaffle(unittest.TestCase):

    def setUp(self):
        self._timeseries = np.random.randint(0, 2, (NROWS*NCOLS), dtype=bool)
    
    def test_make_waffle_array(self):
        
        timeseries = make_waffle_array(self._timeseries, ncols=NCOLS, scale=False)
        self.assertTrue(timeseries.shape[0] == NROWS)
        self.assertTrue(timeseries.shape[1] == NCOLS)



if __name__ == "__main__":
    unittest.main()
