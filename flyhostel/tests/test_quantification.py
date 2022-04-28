import unittest

import argparse
import os.path

import cv2
import numpy as np
import pandas as pd

from flyhostel.quantification.sleep import ethogram_plot_all
from flyhostel.quantification.sleep import load_params
from flyhostel.quantification.sleep import sleep_plot
from flyhostel.quantification.sleep import read_store_metadata

from flyhostel.quantification.sleep import PlottingParams

IMGSTORE_FOLDER = "./tests/static_data/test_imgstore"
ethogram_PLOT_PATH = "./tests/realizations/ethogram_plot.png"
SLEEP_PLOT_PATH_WO_LD = "./tests/realizations/sleep_plot_without_ld.png"
SLEEP_PLOT_PATH_W_LD = "./tests/realizations/sleep_plot_w_ld.png"

CHUNKS = list(range(0, 13))
OFFSET = 58131
ZT0 = 6*3600
START_TIME = 22*3600 + 8 * 60 + 51

class TestQuantificationPlots(unittest.TestCase):

    """
    Test the plotting funcionality of flyhostel.quantification
    A correct ethogram and sleep trace plot should be generated
    """

    def setUp(self):

        datasets = {"data": None, "dt_sleep": None, "dt_binned": None}

        for dataset in datasets:
            csv_file = os.path.join(
                "tests", "static_data", "test_flyhostel", dataset + ".csv"
            )
            datasets[dataset] = pd.read_csv(csv_file, index_col=0)

        self._datasets = datasets


        store_metadata, chunk_metadata = read_store_metadata(
            IMGSTORE_FOLDER, chunk_numbers=CHUNKS
        )

        experiment_name = "test"
        self._analysis_params = argparse.Namespace(
            time_window_length=10,
            velocity_correction_coef=2,
            min_time_immobile=300,
            summary_time_window=30*60, sumary_FUN="mean",
            reference_hour=6,
            offset=OFFSET / 3600
        )

        chunk_index = [self._analysis_params.offset + c / 12 for c in CHUNKS]

        self._plotting_params = argparse.Namespace(
            chunk_index=chunk_index,
            experiment_name="test",
            ld_annotation=False,
            number_of_animals=1
        )

    def test_ethogram_plot(self):

        ethogram_plot = ethogram_plot_all(
            self._datasets["data"],
            self._analysis_params,
            self._plotting_params,
        )
        ethogram_plot.savefig(ethogram_PLOT_PATH)
        # expectation = cv2.imread("static_data/test_flyhostel/ethogram_plot.png")
        realization = cv2.imread(ethogram_PLOT_PATH)
        # self.assertTrue(np.sum(np.abs(expectation - realization)) == 0)


    def test_sleep_plot(self):
        
        fig1 = sleep_plot(
            self._datasets["dt_binned"],
            plotting_params=self._plotting_params
        )

        fig1.savefig(SLEEP_PLOT_PATH_WO_LD)


        self._plotting_params.ld_annotation = True
        fig2 = sleep_plot(
            self._datasets["dt_binned"],
            plotting_params=self._plotting_params
        )

        fig1.savefig(SLEEP_PLOT_PATH_W_LD)



    def tearDown(self):
        # os.remove("realizations/ethogram_plot.png")
        pass

if __name__ == "__main__":
    unittest.main()
