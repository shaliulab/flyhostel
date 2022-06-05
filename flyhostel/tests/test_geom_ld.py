import unittest
import os.path

import pandas as pd
import numpy as np
np.random.seed(1000)
import cv2

import matplotlib.pyplot as plt

from flyhostel.plotting import geom_ld_annotation

NDAYS=3
STATIC_DATA_DIR = "flyhostel/tests/static_data"

data = pd.DataFrame({
    "t": np.linspace(start=0, stop=NDAYS*3600*24, num=NDAYS*24*12)
})
data["L"] = ["T" if (t % (3600*24)) <  3600*12 else "F" for t in data["t"]]
data["asleep"] = 50 + np.random.normal(loc=0, scale=5, size=NDAYS*24*12)

current = os.path.join(STATIC_DATA_DIR, "test_geom_ld_current.png")
target = os.path.join(STATIC_DATA_DIR, "test_geom_ld_target.png")

class TestGeomLD(unittest.TestCase):

    def test_geom_ld(self):
        fig = plt.figure(1, figsize=(10, 7), dpi=90, facecolor="white")
        ax = fig.add_subplot(int("111"))
        geom_ld_annotation(data, ax, yrange=(0, 100))
        ax.plot(data["t"] / 3600, data["asleep"], linewidth=1, c="blue")

        fig.savefig(current)
        current_arr=cv2.imread(current)
        target_arr=cv2.imread(target)
        self.assertTrue(all((current_arr - target_arr).flatten() == 0))




if __name__ == "__main__":
    unittest.main()