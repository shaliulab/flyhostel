import os.path
import logging
import json
import re
import time
import numpy as np
import joblib
import pandas as pd
import h5py

MINS=.5
POSE_DATA=os.environ["POSE_DATA"]

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
    # videos/FlyHostel2/1X/2023-05-23_14-00-00/flyhostel/single_animal/000/000260.mp4.predictions.h5
    tokens = analysis_file.split(os.path.sep)
    flyhostel_id, number_of_animals, date_time, _, _, local_identity, filename = tokens[-7:]
    new_link = os.path.join(directory, f"{flyhostel_id}_{number_of_animals}_{date_time}_{local_identity}", filename)
    print(f"Generating link {analysis_file} -> {new_link}")

    if not dry_run:
        os.makedirs(os.path.dirname(new_link), exist_ok=True)
        if os.path.exists(new_link):
            os.remove(new_link)

        status=0
        # status=impute_body_part(analysis_file, "proboscis", "head")

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


def load_file(file):

    if not os.path.exists(file):
        print(f"{file} does not exist")
        return None, None, None, None

    try:
        with h5py.File(file, 'r') as filehandle:
            node_names = filehandle["node_names"][:]
            node_names=[e.decode() for e in node_names]
            tracks = filehandle["tracks"][:]
            score = filehandle["point_scores"][:]

    except Exception as error:
        logging.warning("Cannot open file %s", file)
        raise error

    
    return node_names, tracks, score, file

def load_files(files, n_jobs=1):

    # files=sorted(files)
    # files_ = []
    # for f in files:
    #     local_identity=int(os.path.basename(os.path.dirname(f)))
    #     chunk=int(os.path.basename(f).split(".")[0])
    #     if local_identity == 0:
    #         print(f"local identity = 0 in chunk {chunk}")
    #         continue
    #     files_.append(f)
    # files=files_


    print(f"{len(files)} files will be loaded")
    Output = joblib.Parallel(n_jobs=n_jobs)(
    # Output = joblib.Parallel(n_jobs=1)(
        joblib.delayed(
            load_file
        )(
           file
        )
        for file in files
    )

    datasets=[]
    point_scores=[]
    previous_node_names=None

    template_dataset = None
    template_score = None

    dataset_count = 0
    for node_names, dataset, score, file in Output:
        if dataset is not None:
            dataset_count += 1
            if template_dataset is None:
                template_dataset = dataset.copy()
                template_dataset[:]=np.nan
                template_score = score.copy()
                template_score[:] = np.nan
            
        datasets.append(dataset)
        point_scores.append(score)
        if node_names is None:
            continue

        if previous_node_names is not None:
            assert all([node_names[i] == previous_node_names[i] for i in range(len(node_names))])
        previous_node_names=node_names
    print(f"{dataset_count} datasets have been loaded")

    missing_frames=0
    for i, _ in enumerate(datasets):
        if datasets[i] is None:
            datasets[i] = template_dataset.copy()
            point_scores[i] = template_score.copy()
            missing_frames+=max(template_dataset.shape)
            

    nframes = sum([dataset.shape[3] for dataset in datasets])
    print(f"{nframes} frames loaded. {missing_frames} frames missing")
    return node_names, datasets, point_scores


def generate_single_file(node_names, datasets, point_scores, files, dest_file):
    # need to populate node_names and tracks
    # tracks has shape 1 x 2 x node_names x timepoints (frames in video)
    # node_names is a dataset with a character array. each name is byte encoded
    node_names_bytes = np.array([name.encode() for name in node_names])
    files_bytes = np.array([f.encode() for f in files])
    
    dataset = np.concatenate(datasets, axis=3)
    point_scores = np.concatenate(point_scores, axis=2)

    with h5py.File(dest_file, 'w') as file:
        node_names_d=file.create_dataset("node_names", (len(node_names), ), dtype='|S12')
        node_names_d[:]=node_names_bytes
        files_bytes_d=file.create_dataset("files", (len(files), ), dtype='|S300')
        files_bytes_d[:]=files_bytes
        
        tracks_d=file.create_dataset("tracks", dataset.shape)
        tracks_d[:]=dataset

        point_scores_d=file.create_dataset("point_scores", point_scores.shape)
        point_scores_d[:]=point_scores

    return dest_file

def parse_number_of_animals(cur):

    cur.execute("SELECT value FROM METADATA  WHERE field  = 'idtrackerai_conf';")
    conf=cur.fetchall()[0][0]
    conf=json.loads(conf.strip())
    number_of_animals=int(conf["_number_of_animals"]["value"])
    return number_of_animals

def infer_analysis_path(basedir, local_identity, chunk, number_of_animals):
    if number_of_animals==1:
        return os.path.join(basedir, "flyhostel", "single_animal", str(0).zfill(3), str(chunk).zfill(6)+".mp4.predictions.h5")
    else:
        return os.path.join(basedir, "flyhostel", "single_animal", str(local_identity).zfill(3), str(chunk).zfill(6)+".mp4.predictions.h5")

def load_concatenation_table(cur, basedir):
    cur.execute("SELECT value FROM METADATA where field ='idtrackerai_conf';")
    conf=cur.fetchone()[0]
    number_of_animals=int(json.loads(conf)["_number_of_animals"]["value"])


    cur.execute("PRAGMA table_info('CONCATENATION');")
    header=[row[1] for row in cur.fetchall()]

    cur.execute("SELECT * FROM CONCATENATION;")
    records=cur.fetchall()
    concatenation=pd.DataFrame.from_records(records, columns=header)
    concatenation["dfile"] = [
        infer_analysis_path(basedir, row["local_identity"], row["chunk"], number_of_animals=number_of_animals)
        for i, row in concatenation.iterrows()
    ]
    return concatenation


def pipeline(experiment_name, identity, concatenation, chunks=None):
    if chunks is not None:
        concatenation=concatenation.loc[concatenation["chunk"].isin(chunks)]

    if "1X" in experiment_name:
        concatenation_i=concatenation
    else:
        concatenation_i=concatenation.loc[concatenation["identity"]==identity]

    if chunks is not None:
        assert concatenation_i.shape[0] == len(chunks)

    files=concatenation_i["dfile"]
    node_names, datasets, point_scores = load_files(files)
    dest_file=os.path.join(POSE_DATA, f"{experiment_name}__{str(identity).zfill(2)}", f"{experiment_name}__{str(identity).zfill(2)}.h5")
    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
    generate_single_file(node_names, datasets, point_scores, files, dest_file=dest_file)
    assert os.path.exists(dest_file)

