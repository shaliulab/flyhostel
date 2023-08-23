import warnings
import os
import shutil
import sqlite3
import numpy as np
from flyhostel.data.interactions.load_data import get_sqlite_file
from sleap.io.dataset import Labels
# from sleap.instance import LabeledFrame

FLYHOSTEL_VIDEOS=os.environ["FLYHOSTEL_VIDEOS"]


def get_local_identity(animal, frame_number):

    identity = int(animal.split("__")[1])
    sqlite_file = get_sqlite_file(animal)

    with sqlite3.connect(sqlite_file) as conn:

        cursor = conn.cursor()
        cursor.execute("SELECT identity, local_identity FROM IDENTITY WHERE frame_number = ?;", (frame_number,))
        rows = cursor.fetchall()
        rows = [row for row in rows if row[0] == identity]
        if rows:
            row=rows[0]
            local_identity = int(float(row[1]))
        else:
            local_identity=None
        
    return local_identity

def get_experiment_dir(animal):

    experiment_dir = animal.split("__")[0]
    tokens = experiment_dir.split("_")
    experiment_dir = f"{tokens[0]}/{tokens[1]}/{tokens[2]}_{tokens[3]}/"
    return experiment_dir


stride=15
def generate_sleap_files(interactions, root):

    interactions.sort_values(["animal0", "animal1"], inplace=True)
    
    files=[]
    
    for interaction_id, interaction in interactions.iterrows():

        chunk = interaction["chunk"]
        frame_number = interaction["frame_number"]
        animal0=interaction["animal0"]
        animal1=interaction["animal1"]
        animals=[animal0, animal1]
        for animal in animals:
            labeled_frames=[]


            local_identity=get_local_identity(animal, frame_number)
            if local_identity is None:
                warnings.warn(f"Animal {animal} not found in frame_number {frame_number}")
                continue
    
            experiment_dir = get_experiment_dir(animal)
            subfolder=f"{FLYHOSTEL_VIDEOS}/{experiment_dir}/flyhostel/single_animal/{str(local_identity).zfill(3)}/"
            mp4_file = f"{subfolder}/{str(chunk).zfill(6)}.mp4"
            sleap_file = f"{mp4_file}.predictions.slp"
            key=f"{animal}_{str(chunk).zfill(6)}_{str(local_identity).zfill(3)}"
            

            wd=os.getcwd()
            os.chdir(subfolder)
            print(f"Loading {sleap_file}")
            labels = Labels.load_file(sleap_file)
            print("Done!")
            os.chdir(wd)

            assert len(labels.videos)==1
            video=labels.videos[0]
            
            dest_mp4_filemame=f"{key}.mp4"
            video.backend.filename=dest_mp4_filemame
                       

            skeletons=labels.skeletons
            selected_frames=np.arange(interaction["frame_idx"], interaction["frame_idx"]+interaction["length"]*stride)

            for lf in labels.labeled_frames:
                if lf.frame_idx in selected_frames:
                    labeled_frames.append(lf)

            new_labels=Labels(labeled_frames=labeled_frames, videos=[video], skeletons=skeletons)

            filename=f"{root}/{key}/{animal0}_{animal1}___{str(chunk).zfill(6)}.slp"
            dest_mp4=f"{root}/{key}/{dest_mp4_filemame}"

            os.makedirs(os.path.dirname(dest_mp4), exist_ok=True)
            shutil.copy(mp4_file, dest_mp4)
            new_labels.save_file(labels=new_labels, filename=filename)
            files.append(filename)

    return files
