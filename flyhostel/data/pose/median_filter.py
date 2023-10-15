import pandas as pd
def median_filter(h5s_pandas, window=5, center=True, **kwargs):
    out = []
    for h5 in h5s_pandas:
        h5.loc[:, pd.IndexSlice[:,:, ["x", "y"]]]=h5.loc[:, pd.IndexSlice[:,:, ["x", "y"]]].rolling(
            window=window, center=center, **kwargs
        ).median()
        out.append(h5)
    
    return out