import os.path
import logging
import json
import re
import time
import h5py
import sqlite3
import numpy as np
import joblib
import pandas as pd
from tqdm.auto import tqdm
from .cvat import experiment_is_validated
from .utils import get_basedir, get_chunksize, get_dbfile, get_number_of_animals
MINS=.5

def find_files(directory, pattern):
    hits=[]
    regex = re.compile(pattern)
    for root, dirs, files in os.walk(directory):
        for file in files:
            if regex.match(file):
                hits.append(os.path.join(root, file))

    return hits

def file_is_older_than(path, mins):
    timestamp=os.path.getmtime(path)
    now = time.time()
    age = now-timestamp

    return age, age > (mins*60)


def make_link(analysis_file, directory, dry_run=False):

    age, older = file_is_older_than(analysis_file, MINS)

    if not older:
        print(f"Skipping {analysis_file}, age {age} < {MINS} mins")
        return

    assert os.path.isdir(directory)
    tokens = analysis_file.split(os.path.sep)
    flyhostel_id, number_of_animals, date_time, _, _, local_identity, filename = tokens[-7:]
    new_link = os.path.join(directory, f"{flyhostel_id}_{number_of_animals}_{date_time}_{local_identity}", filename)
    print(f"Generating link {analysis_file} -> {new_link}")

    if not dry_run:
        os.makedirs(os.path.dirname(new_link), exist_ok=True)
        if os.path.exists(new_link):
            os.remove(new_link)

        status=0

        if status is None:
            return
        
        os.symlink(analysis_file, new_link)

def impute_body_part(analysis_file, body_part, reference):

    with h5py.File(analysis_file, "a") as filehandle:

        if "imputation" in filehandle.keys():
            return np.array([])

        try:
            node_names=[element.decode() for element in filehandle["node_names"][:]]
        except:
            return None
        bp_index=node_names.index(body_part)
        ref_index=node_names.index(reference)
        missing = np.isnan(filehandle["tracks"][:, :, bp_index])[0, 0]

        ref_not_missing = np.bitwise_not(np.isnan(filehandle["tracks"][:, :, ref_index])[0, 0])
        indexer = np.bitwise_and(missing, ref_not_missing)
        filehandle["tracks"][:, :, bp_index, indexer] = filehandle["tracks"][:, :, ref_index, indexer]
        imputation=filehandle.create_dataset("imputation", (indexer.shape[0],), dtype='bool')
        imputation[:]=indexer

    return missing


def load_file(file, chunksize=None):
    if chunksize is None:
        print("chunksize is not passed. No checks for missing data will be made")
    if not os.path.exists(file):
        print(f"{file} does not exist")
        return None, None, None, None, None            # <-- 5 now

    try:
        with h5py.File(file, 'r') as filehandle:
            node_names = [e.decode() for e in filehandle["node_names"][:]]
            tracks = filehandle["tracks"][:]
            score = filehandle["point_scores"][:]
            # per-instance, per-frame confidence. Older exports may lack it.
            if "instance_scores" in filehandle:
                inst = filehandle["instance_scores"][:]          # (n_tracks, n_frames)
            else:
                inst = np.full((tracks.shape[0], tracks.shape[3]), np.nan)
                logging.warning("%s has no instance_scores -> filled NaN", file)
    except Exception as error:
        logging.warning("Cannot open file %s", file)
        raise error

    if chunksize is not None:
        assert tracks.shape[3] == chunksize, \
            f"{file} is missing pose estimates (found {tracks.shape[3]} instead of {chunksize})"

    return node_names, tracks, score, inst, file          # <-- inst added


def load_files(files, chunksize, n_jobs=1):
    print(f"{len(files)} files will be loaded")
    Output = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(load_file)(file, chunksize=chunksize) for file in files
    )

    datasets, point_scores, inst_scores = [], [], []
    previous_node_names = None
    template_dataset = template_score = template_inst = None
    dataset_count = 0

    for i, (node_names, dataset, score, inst, file) in enumerate(Output):   # <-- inst
        if dataset is not None:
            dataset_count += 1
            if template_dataset is None:
                template_dataset = np.full_like(dataset, np.nan)
                template_score = np.full_like(score, np.nan)
                template_inst = np.full_like(inst, np.nan)                    # <-- template
        else:
            raise ValueError(f"{file} could not be loaded")

        datasets.append(dataset)
        point_scores.append(score)
        inst_scores.append(inst)                                             # <-- collect

        if node_names is None:
            continue
        if previous_node_names is not None:
            assert all(node_names[j] == previous_node_names[j] for j in range(len(node_names)))
        previous_node_names = node_names
    print(f"{dataset_count} datasets have been loaded")

    missing_frames = 0
    for i, _ in enumerate(datasets):
        if datasets[i] is None:
            datasets[i] = template_dataset.copy()
            point_scores[i] = template_score.copy()
            inst_scores[i] = template_inst.copy()                            # <-- fill
            missing_frames += max(template_dataset.shape)

    nframes = sum(ds.shape[3] for ds in datasets)
    print(f"{nframes} frames loaded. {missing_frames} frames missing")
    return node_names, datasets, point_scores, inst_scores                   # <-- 4 returns


def generate_single_file(node_names, datasets, point_scores, inst_scores, files, dest_file):
    node_names_bytes = np.array([name.encode() for name in node_names])
    files_bytes = np.array([f.encode() for f in files])

    point_scores = np.concatenate(point_scores, axis=2)   # (n_tracks, n_nodes, frames)
    inst_scores  = np.concatenate(inst_scores,  axis=1)   # (n_tracks, frames)  <-- axis=1

    track_names = ["individual_0"]
    total_frames = sum(ds.shape[3] for ds in datasets)
    final_shape = (datasets[0].shape[0], datasets[0].shape[1], len(node_names), total_frames)
    chunksize = datasets[0].shape[3]
    print(f"Chunksize = {chunksize}. Final shape = {final_shape}")

    with h5py.File(dest_file, 'w') as file:
        node_names_d = file.create_dataset("node_names", (len(node_names),), dtype='|S12')
        node_names_d[:] = node_names_bytes
        files_bytes_d = file.create_dataset("files", (len(files),), dtype='|S300')
        files_bytes_d[:] = files_bytes

        tracks_d = file.create_dataset(
            "tracks", shape=final_shape,
            chunks=(1, 2, len(node_names), chunksize),
            compression="gzip", compression_opts=4)
        offset = 0
        print(f"Writing pose to disk -> {dest_file}")
        for dataset in datasets:
            n = dataset.shape[3]
            tracks_d[:, :, :, offset:offset + n] = dataset
            offset += n
        print("Done")

        tn = file.create_dataset("track_names", (len(track_names),), dtype="|S12")
        tn[:] = np.array([e.encode() for e in track_names])

        ps = file.create_dataset("point_scores", point_scores.shape)
        ps[:] = point_scores

        isd = file.create_dataset("instance_scores", inst_scores.shape)      # <-- new
        isd[:] = inst_scores

        return dest_file


def parse_number_of_animals(cur):

    cur.execute("SELECT value FROM METADATA  WHERE field  = 'idtrackerai_conf';")
    conf=cur.fetchall()[0][0]
    conf=json.loads(conf.strip())
    number_of_animals=int(conf["_number_of_animals"]["value"])
    return number_of_animals


def pipeline(experiment_name, identity, concatenation, chunks=None, output=".", strict=True, n_jobs=1):

    chunksize=get_chunksize(experiment_name)

    if chunks is not None:
        concatenation=concatenation.loc[concatenation["chunk"].isin(chunks)]

    if "1X" in experiment_name:
        concatenation_i=concatenation
        chunk, count = np.unique(concatenation["chunk"], return_counts=True)
        if not all(count==1):
            bad_chunks = chunk[count!=1]
            raise Exception(f"More than 1 animal found in a single animal experiment. Chunks {bad_chunks}")
        
        identity=0

    else:
        concatenation_i=concatenation.loc[concatenation["identity"]==identity]

    if chunks is not None:
        if concatenation_i.shape[0] < len(chunks):
            print(f"{concatenation_i.shape[0]} < {len(chunks)}. The concatenation is missing data")
            if strict:
                raise Exception(f"Chunks missing in concatenation table for identity {identity}: {set(chunks).difference(set(concatenation_i['chunk'].tolist()))}")
            else:
                first_chunk_missing=concatenation_i.iloc[1:].loc[concatenation_i["chunk"].diff().iloc[1:]!=1]["chunk"].iloc[0]
                concatenation_i=concatenation_i.query(f"chunk < {first_chunk_missing}")



    files=concatenation_i["dfile"]

    if n_jobs is None:
        n_jobs=-1

    node_names, datasets, point_scores, inst_scores = load_files(files, chunksize, n_jobs=n_jobs)
    

    dest_file=os.path.join(output, f"{experiment_name}__{str(identity).zfill(2)}", f"{experiment_name}__{str(identity).zfill(2)}.h5")
    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
    
    generate_single_file(node_names, datasets, point_scores, inst_scores, files, dest_file=dest_file)

    assert os.path.exists(dest_file)


def infer_analysis_path(basedir, local_identity, chunk, number_of_animals):
    if number_of_animals==1:
        return os.path.join(basedir, "flyhostel", "single_animal", "000",                        str(chunk).zfill(6)+".mp4.predictions.h5")
    else:
        return os.path.join(basedir, "flyhostel", "single_animal", str(local_identity).zfill(3), str(chunk).zfill(6)+".mp4.predictions.h5")


def load_concatenation_table(cur, basedir, concatenation_table="CONCATENATION_VAL", errors="raise"):
    cur.execute("SELECT value FROM METADATA where field ='idtrackerai_conf';")
    conf=cur.fetchone()[0]
    number_of_animals=int(json.loads(conf)["_number_of_animals"]["value"])

    cur.execute(f"PRAGMA table_info('{concatenation_table}');")
    header=[row[1] for row in cur.fetchall()]

    cur.execute(f"SELECT * FROM {concatenation_table};")
    records=cur.fetchall()
    concatenation=pd.DataFrame.from_records(records, columns=header)
    concatenation["chunk"]=concatenation["chunk"].astype(int)
    
    concatenation.sort_values("chunk", inplace=True)
    diff=concatenation["chunk"].drop_duplicates().diff().iloc[1:]

    
    if errors == "raise":
        if not (diff==1).all():
            rows=np.where(diff!=1)[0].tolist()
            rows=sorted(rows + (np.array(rows)+1).tolist())
            print(concatenation.iloc[1:].loc[sorted(rows)])
            raise ValueError("Missing chunks in concatenation")

    concatenation["dfile"] = [
        infer_analysis_path(basedir, int(row["local_identity"]), str(int(row["chunk"])).zfill(6), number_of_animals=number_of_animals)
        for i, row in concatenation.iterrows()
    ]
    return concatenation

def recreate_pose_file(experiment, identity, chunks=None, output=".", n_jobs=1):
    """
    Arguments
        experiment (str): Identifier, the basedir and dbfile are derived from there
        chunks (list): Can be left None to infer from found chunkwise pose files
        output (str): Path to folder where the file will be saved
        identity (int)
        n_jobs (int): Parallel processing in case more than one identity is requested
    
    NOTE: Multiple identities are not tested properly. Recommended to pass just one
    """

    basedir = get_basedir(experiment)
    dbfile = get_dbfile(basedir)
    number_of_animals = get_number_of_animals(experiment)

    with sqlite3.connect(dbfile) as conn:
        cur=conn.cursor()
        if number_of_animals > 1 and experiment_is_validated(experiment, errors="raise"):
            concatenation_table="CONCATENATION_VAL"
        else:
            concatenation_table="CONCATENATION"

        concatenation=load_concatenation_table(cur, basedir, concatenation_table=concatenation_table)

    if identity is None:
        identities=range(1, number_of_animals+1)
    else:
        identities=[int(identity)]

    # joblib.Parallel(n_jobs=n_jobs)(
    #     joblib.delayed(
    #         pipeline
    #     )(
    #         experiment, identity, concatenation, chunks, output=output, strict=False #, write_only=args.write_only
    #     )
    #     for identity in identities
    # )
    
    for identity in identities:
        pipeline(experiment, identity, concatenation, chunks, output=output, strict=False, n_jobs=n_jobs #, write_only=args.write_only
    )

    return None


def get_pose_file(experiment, identity, pose_name, recreate=True):
    animal=experiment + "__" + str(identity).zfill(2)
    basedir=get_basedir(experiment)
    pose_file=os.path.join(
        basedir, "motionmapper",
        str(identity).zfill(2),
        f"pose_{pose_name}",
        animal,
        animal + ".h5"
    )
    if not os.path.exists(pose_file) and pose_name=="raw" and recreate:
        output=os.path.dirname(os.path.dirname(pose_file))
        recreate_pose_file(experiment=experiment, identity=identity, chunks=None, output=output, n_jobs=None)

    return pose_file