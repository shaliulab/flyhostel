from umap import UMAP
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler

def run_dr(df, features, scaler, model, fit=True, name="UMAP"):
    print(features)
    X=df[features]
    missing=X.isna().any(axis=1)
    is_infinite=np.isinf(X).any(axis=1)
    X=X[~(missing|is_infinite)]
    if fit:
        X_norm=scaler.fit_transform(X)
        proj=model.fit_transform(X_norm)

    else:
        X_norm=scaler.transform(X)
        proj=model.predict(X_norm)

    for i in range(proj.shape[1]):
        df[f"{name}{i+1}"]=np.nan
        df.loc[~(missing|is_infinite), f"{name}{i+1}"]=proj[:,i]

    for i, feat in enumerate(features):
        df.loc[~(missing|is_infinite), feat + "_norm"]=X_norm[:,i]
        
    return df


def main(df, features, algorithm="UMAP"):
    if algorithm =="PCA":
        model=PCA()
    elif algorithm=="UMAP":
        model=UMAP()
    else:
        raise NotImplementedError(f"Algorithm {algorithm} not supported")
    scaler=StandardScaler()
    df=run_dr(df, features, scaler, model, fit=True, name=algorithm)
    return df
