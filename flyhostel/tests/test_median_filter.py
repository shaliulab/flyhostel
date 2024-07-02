import numpy as np
import pandas as pd
from flyhostel.data.pose.median_filter import median_filter

def test_median_filter():

    h5s=[]
    for body_part in ["bp1", "bp2"]:
        h5 = pd.DataFrame({"frame_number": np.arange(0, 30*15, 15), "x": np.random.randint(0, 100, 30), "y": np.random.randint(0, 100, 30)})
        h5.set_index("frame_number", inplace=True)
        h5.columns = pd.MultiIndex.from_tuples([
            ("SLEAP", body_part, "x"),
            ("SLEAP", body_part, "y"),
        ])

        h5s.append(h5)

    h5 = pd.merge(h5s[0], h5s[1],left_index=True, right_index=True)
    h5s=[h5, h5]
    h5s = median_filter(h5s)

    print(h5s)


test_median_filter()
