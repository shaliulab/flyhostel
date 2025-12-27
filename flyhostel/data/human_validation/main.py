"""
Generate short videos whenever the tracking analysis captured in the dbfile
has issues
"""
import os.path
import logging

import pandas as pd
import joblib

from flyhostel.utils.utils import (
    get_framerate,
    get_first_frame,
    get_last_frame,
    get_dbfile,
    get_chunksize,
)
from flyhostel.data.human_validation.utils import load_tracking_data, FIELD
from flyhostel.data.human_validation.qc import analyze_video
from flyhostel.data.human_validation.video import generate_validation_video
from zeitgeber.rle import encode

logger=logging.getLogger(__name__)


def annotate_for_validation(
        experiment, output_folder,
        time_window_length=1,
        format=".png", n_jobs=20,
        min_frame_number=None,
        max_frame_number=None,
        cache=False,
    ):
    """
    Entrypoint make-identogram

    Generate movies capturing each scene where validation may be needed based on heuristics and QC
    """

    tokens=experiment.split("_")
    suffix="/".join([tokens[0], tokens[1], "_".join(tokens[2:4])])

    number_of_animals=int(tokens[1].replace("X",""))

    basedir=os.path.join(os.environ["FLYHOSTEL_VIDEOS"], suffix)
    store_path=os.path.join(basedir, "metadata.yaml")
    dbfile=os.path.join(basedir, experiment + ".db")
    assert os.path.exists(dbfile), f"{dbfile} not found"

    output_path_feather_bin=os.path.join(output_folder, experiment + f"_machine-validation-index-{time_window_length}-s.feather")
    output_path_csv=os.path.join(output_folder, experiment + f"_machine-qc-index-{time_window_length}-s.csv")

    df_bin=None
    df=None
    qc_fail=None

    dbfile = get_dbfile(basedir)
    chunksize = get_chunksize(experiment)
    framerate = get_framerate(experiment)

    if min_frame_number is None:
        min_frame_number=get_first_frame(dbfile)

    if max_frame_number is None:
        max_frame_number=get_last_frame(dbfile)
        
    
    df=load_tracking_data(
        dbfile, output_folder, experiment,
        min_frame_number=min_frame_number,
        max_frame_number=max_frame_number,
        n_jobs=n_jobs, cache=cache
    )

    if os.path.exists(output_path_feather_bin) and cache:
        df_bin=pd.read_feather(output_path_feather_bin)
        if min_frame_number is not None:
            df_bin=df_bin.loc[
                (df_bin["frame_number"] >= min_frame_number)
            ]

        if max_frame_number is not None:
            df_bin=df_bin.loc[
                (df_bin["frame_number"] < max_frame_number)
            ]
    else:
        
        # logger.debug("Binning into %s second windows", time_window_length)
        df_bin=df.copy()
        df_bin["t_round"]=df_bin["t"].copy()
        # df_bin=bin_windows(df, time_window_length=time_window_length)

    logger.debug("Running QC of experiment %s", experiment)

    qc, tests=analyze_video(
        df_bin.copy(), number_of_animals,
        min_frame_number=min_frame_number,
        max_frame_number=max_frame_number,
        chunksize = chunksize,
        n_jobs=n_jobs,
    )
    logger.info("%s %% of %s passes QC", round(100*qc["qc"].mean(), 2), experiment)

    df_bin=df_bin.drop(tests, axis=1, errors="ignore").merge(qc[["frame_number", "chunk"] + tests], on=["chunk", "frame_number"], how="left")

    qc_rle=pd.DataFrame.from_records(encode([str(e)[0] for e in qc["qc"].values]), columns=["status", "length"])
    qc_rle["duration"]=qc_rle["length"]*time_window_length
    qc_rle["experiment"]=experiment


    qc_rle["index"]=[0] + qc_rle["length"].cumsum().tolist()[:-1]
    qc_rle["frame_number"]=qc["frame_number"].iloc[qc_rle["index"]].values
    qc_rle["chunk"]=qc["chunk"].iloc[qc_rle["index"]].values

    for qc_col in tests:
        qc_rle[qc_col]=qc[qc_col].iloc[qc_rle["index"]].values
        
    qc_fail=qc_rle.loc[qc_rle["status"]=="F"]
    
    if cache and not os.path.exists(output_path_feather_bin):
        df_bin.to_feather(output_path_feather_bin)
    
    
    margin_size=1
    qc_fail["last_frame_number"]=qc_fail["frame_number"]+qc_fail["length"]+margin_size
    qc_fail["frame_number"]-=margin_size    
    qc_fail.to_csv(output_path_csv)

    kwargs=[]

    for i, row in qc_fail.iterrows():
        frame_number_0=row["frame_number"]-margin_size
        frame_number_last=row["last_frame_number"]+margin_size
        tracking_data=df.loc[
            (df["frame_number"]>=frame_number_0) &
            (df["frame_number"]<=frame_number_last)
        ]
        if df.shape[0]>0:
            kwargs.append({"row": row, "tracking_data": tracking_data})

    logger.debug("Will generate %s videos", len(kwargs))

    movies_folder=os.path.join(output_folder, "movies")
    os.makedirs(movies_folder, exist_ok=True)

    joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(
            generate_validation_video
        )(
            store_path, row=kwargs[i]["row"], df=kwargs[i]["tracking_data"],
            number_of_animals=number_of_animals, output_folder=movies_folder,
            chunksize=chunksize, framerate=framerate, format=format, field=FIELD
        )
        for i in range(len(kwargs))
    )
    return df, df_bin, qc_fail


def bin_windows(df, time_window_length=1):
    """
    Bin the tracking results into brief bins. Every unique entry of local identity identity chunk fragment or modified
    gets a separate bin
    """
    df["t_round"]=time_window_length*(df["t"]//time_window_length)
    # df["t_round"]=df["t"].copy()

    df_bin=df.drop_duplicates(["t_round","local_identity", "identity", "chunk", "fragment", "modified"])[[
        "t_round","local_identity", "identity", "chunk", "fragment", "modified", "x", "y"
    ]]

    df_bin=df_bin.merge(
        df[["t_round", "frame_number", "chunk", "frame_idx"]].groupby(["t_round", "chunk"]).first().reset_index(),
        on=["t_round", "chunk"],
        how="left"
    )
    df_bin.sort_values(["chunk", "frame_number", "local_identity"], inplace=True)
    return df_bin


