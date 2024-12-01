import umap
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler

MODELS={"UMAP": umap.UMAP, "PCA": PCA}


def run_dr(df, features, scaler=None, model=None, fit=True, name="UMAP"):
    print(features)
    X=df[features]
    missing=X.isna().any(axis=1)
    is_infinite=np.isinf(X).any(axis=1)
    X=X[~(missing|is_infinite)]

    if scaler is None:
        scaler=StandardScaler()


    if model is None:
        model=MODELS[name]()

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
        
    return df, model, scaler


def main(df, features, algorithm="UMAP"):
    if algorithm in MODELS:
        model=MODELS[algorithm]()
    else:
        raise NotImplementedError(f"Algorithm {algorithm} not supported")
    scaler=StandardScaler()
    df, model, scaler=run_dr(df, features, scaler, model, fit=True, name=algorithm)
    return df, model, scaler
