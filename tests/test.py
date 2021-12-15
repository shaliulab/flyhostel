import unittest
import os.path

import cv2
import numpy as np
import pandas as pd

from flyhostel.quantification.sleep import waffle_plot_all
from flyhostel.quantification.sleep import load_params
from flyhostel.quantification.sleep import read_store_metadata

from flyhostel.quantification.sleep import PlottingParams

IMGSTORE_FOLDER = "./tests/static_data/test_imgstore"
WAFFLE_PLOT_PATH = "realizations/waffle_plot.png"

class TestQuantificationPlots(unittest.TestCase):

    """
    Test the plotting funcionality of flyhostel.quantification
    A correct waffle and sleep trace plot should be generated
    """
    
    
    def setUp(self):

        datasets = {"data", "dt_sleep", "dt_binned"}
        for dataset in datasets:
            csv_file = os.path.join("tests", "static_data", "test_flyhostel", dataset + ".csv")
            datasets[dataset] = pd.read_csv(csv_file, index_col=0)
        
        self._datasets = datasets
        
        CHUNKS = list(range(0, 12))

        store_metadata, chunk_metadata = read_store_metadata(
            IMGSTORE_FOLDER, chunk_numbers=CHUNKS
        )

        experiment_name="test_imgstore"
        self._analysis_params, self._plotting_params  = load_params(store_metadata, chunks=CHUNKS, experiment_name=experiment_name)

    
    def test_waffle_plot(self):
        
        waffle_plot = waffle_plot_all(self._datasets["data"], self._analysis_params, self._plotting_params)
        waffle_plot.savefig(WAFFLE_PLOT_PATH)
        # expectation = cv2.imread("static_data/test_flyhostel/waffle_plot.png")
        realization = cv2.imread(WAFFLE_PLOT_PATH)
        # self.assertTrue(np.sum(np.abs(expectation - realization)) == 0)

    def tearDown(self):
        # os.remove("realizations/waffle_plot.png")
        pass


if __name__ == '__main__':
    unittest.main()
