"""
Generate short videos whenever the tracking analysis captured in the dbfile
has issues
"""
import os.path
import logging
import joblib

from flyhostel.utils.utils import (
    get_framerate,
    get_basedir,
    get_first_frame,
    get_last_frame,
    get_number_of_animals,
    get_dbfile,
    get_chunksize,
)
from flyhostel.data.human_validation.utils import load_tracking_data, FIELD
from flyhostel.data.human_validation.qc import analyze_experiment
from flyhostel.data.human_validation.video import generate_validation_video
from flyhostel.data.pose.ethogram.utils import annotate_bouts, annotate_bout_duration

logger=logging.getLogger(__name__)

TESTING=False


def annotate_for_validation(
        experiment,
        output_folder,
        format,
        time_window_length=1,
        n_jobs=20,
        min_frame_number=None,
        max_frame_number=None,
        cache=False,
    ):
    """
    Generate movies capturing each scene where validation may be needed based on heuristics and QC
    It is the entrypoint of make-identogram

    QC is performed in analyze_experiment
    Then frames with some QC problem are saved as videos,
        one video for every group of consecutive frames with problems
    """


    basedir=get_basedir(experiment)
    number_of_animals=get_number_of_animals(experiment)
    dbfile=get_dbfile(basedir)
    assert os.path.exists(dbfile), f"{dbfile} not found"
    chunksize = get_chunksize(experiment)
    framerate = get_framerate(experiment)

    output_path_csv=os.path.join(output_folder, experiment + f"_machine-qc-index-{time_window_length}-s.csv")

    df_bin=None
    df=None
    qc_fail=None

    if min_frame_number is None:
        min_frame_number=get_first_frame(dbfile)

    if max_frame_number is None:
        max_frame_number=get_last_frame(dbfile)
        
    
    df=load_tracking_data(
        experiment, output_folder,
        min_frame_number=min_frame_number,
        max_frame_number=max_frame_number,
        n_jobs=n_jobs, cache=cache
    )
    df["t_round"]=df["t"].copy()

    logger.debug("Running QC of experiment %s", experiment)

    qc=analyze_experiment(
        df.copy(),
        number_of_animals=number_of_animals,
        chunksize=chunksize,
        n_jobs=n_jobs,
    )
    qcs=[col for col in qc.columns if col != "frame_number"]
    qc["chunk"]=qc["frame_number"]//chunksize

    logger.info("%s %% of %s passes QC", round(100*qc["qc"].mean(), 2), experiment)

    # TODO
    # If qc can be used instead of df in annotate_bouts
    # this block can be simplified like this
    #######
    df=df.merge(
        qc[["frame_number"] + qcs], on="frame_number", how="left"
    )
    qc_rle = df[["frame_number"] + qcs]
    #######
    # qc_rle = annotate_bout_duration(annotate_bouts(qc.copy(), variable="qc"), fps=framerate)\
    #######

    qc_rle = annotate_bout_duration(annotate_bouts(df.copy(), variable="qc"), fps=framerate)\
        .query("bout_in==1")\
        .rename({"bout_out": "length", "qc": "status"}, axis=1)

    qc_fail=qc_rle.loc[qc_rle["status"]=="F"].copy()

    margin_size=2
    qc_fail["last_frame_number"]=qc_fail["frame_number"]+qc_fail["length"]+margin_size
    qc_fail["frame_number"]-=margin_size
    qc_fail.to_csv(output_path_csv)

    data=[]
    for i, row in qc_fail.iterrows():
        frame_number_0=row["frame_number"]
        frame_number_last=row["last_frame_number"]
        tracking_data=df.loc[
            (df["frame_number"]>=frame_number_0) &
            (df["frame_number"]<=frame_number_last)
        ]
        if df.shape[0]>0:
            data.append({"row": row, "tracking_data": tracking_data})

    logger.debug("Will generate %s videos", len(data))

    if TESTING:
        data=data[:3]

    movies_folder=os.path.join(output_folder, "movies")
    os.makedirs(movies_folder, exist_ok=True)
    
    # these could be fetched inside generate_validation_video
    # but we save lots of lookups to the dbfile this way
    kwargs=dict(chunksize=chunksize, framerate=framerate, number_of_animals=number_of_animals)
    joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(
            generate_validation_video
        )(
            experiment,
            row=data[i]["row"],
            df=data[i]["tracking_data"],
            output_folder=movies_folder,
            format=format,
            field=FIELD,
            **kwargs
        )
        for i in range(len(data))
    )
    return df, qc_fail


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
