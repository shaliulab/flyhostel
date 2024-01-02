import warnings
import os
import shutil
import sqlite3
import numpy as np
from flyhostel.data.interactions.load_data import get_sqlite_file
try:
    import sleap
    from sleap.io.dataset import Labels
    TRACK=sleap.io.dataset.Track(name="Track-0", spawned_on=0)

except Exception as error:
    sleap=None
    Labels=None
    TRACK=None
    print(f"SLEAP cannot be loaded. SLEAP integration disabled")
    print(error)

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



def merge_labels(labels1, labels2):
    """
    Combine labels stored in two different label objects

    Instances without track get assigned the track Track-0 spawned on frame 0

    Based on https://colab.research.google.com/drive/1OmnmZIM64pWFqoeNkUm2QnpOWmPwoUbN?usp=sharing#scrollTo=52ojKlgcz6ns
    """
    if sleap is None:
        return None
    
    nodes1_inds = np.array([labels1.skeleton.node_names.index(node) for node in labels2.skeleton.node_names])
    
    for src_lf in labels2:
        if (src_lf.video, src_lf.frame_idx) in labels1:
            # If the video/frame_idx is labeled in the base dataset, we'll just add to it
            dst_lf = labels1[(src_lf.video, src_lf.frame_idx)]

        else:
            # Create a new labeled frame and add it to the base dataset
            dst_lf = sleap.LabeledFrame(video=src_lf.video, frame_idx=src_lf.frame_idx)
            labels1.append(dst_lf)

        for inst in src_lf:

            if inst.track is None:
                track = TRACK
            else:
                track = inst.track
        
            # Create new instance with unified skeleton
            if type(inst) == sleap.PredictedInstance:
                old_pts = inst.numpy()
                old_scores = inst.scores

                new_pts = np.full((len(labels1.skeleton), 2), np.nan)
                new_pts[nodes1_inds] = old_pts
                new_scores = np.full((len(labels1.skeleton)), np.nan)
                new_scores[nodes1_inds] = old_scores

                new_inst = sleap.PredictedInstance.from_numpy(new_pts, new_scores, instance_score=inst.score, skeleton=labels1.skeleton, track=track)

            elif type(inst) == sleap.Instance:
                old_pts = inst.numpy()

                new_pts = np.full((len(labels1.skeleton), 2), np.nan)
                new_pts[nodes1_inds] = old_pts

                new_inst = sleap.Instance.from_numpy(new_pts, skeleton=labels1.skeleton, track=track)

            # TODO: Handle 100% overlapping instances to prevent trivial duplicates

            # Add the new instance to the destination frame in the base labels
            labels1.add_instance(dst_lf, new_inst)

            if new_inst.track not in labels1.tracks:
                # TODO: Rename tracks with the duplicate names to avoid confusion
                labels1.tracks.append(new_inst.track)


    return labels1


stride=15
def generate_sleap_files(interactions, root):

    interactions.sort_values(["animal0", "animal1"], inplace=True)
    
    files=[]

    all_animals=np.unique(interactions["animal0"]).tolist()
    all_animals.extend(np.unique(interactions["animal1"]).tolist())

    assert len(all_animals) == 2, f"Interactions between more than 2 animals not supported"
    labeled_frames={animal: [] for animal in all_animals}
    videos={animal: [] for animal in all_animals}
    skeletons={animal: [] for animal in all_animals}
    labels_by_animal={animal: [] for animal in all_animals}

    for interaction_id, interaction in interactions.iterrows():

        chunk = interaction["chunk"]
        frame_number = interaction["frame_number"]
        animal0=interaction["animal0"]
        animal1=interaction["animal1"]
        animals=[animal0, animal1]
        for animal in animals:

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
            
            dest_mp4_filemame=f"SLEAP_DATA/{key}/{key}.mp4"
            video.backend.filename=dest_mp4_filemame
    
            selected_frames=np.arange(interaction["frame_idx"], interaction["frame_idx"]+interaction["length"]*stride)

            labeled_frames_local=[]
            for lf in labels.labeled_frames:
                if lf.frame_idx in selected_frames:
                    labeled_frames_local.append(lf)

            new_labels = Labels(labeled_frames=labeled_frames_local, videos=[video], skeletons=labels.skeletons)

            labels_by_animal[animal].append(
                new_labels
            )

            dest_mp4=f"{root}/{dest_mp4_filemame}"

            os.makedirs(os.path.dirname(dest_mp4), exist_ok=True)
            shutil.copy(mp4_file, dest_mp4)


    for animal in all_animals:
        other_animal = all_animals.copy()
        other_animal.pop(all_animals.index(animal))
        other_animal=other_animal[0]

        for label_object in labels_by_animal[animal][1:]:
            merge_labels(labels_by_animal[animal][0],label_object)

        filename=f"{root}/{animal}_{other_animal}.slp"
        
        print(f"Saving ---> {filename}")
        new_labels.save_file(labels=labels_by_animal[animal][0], filename=filename)
        files.append(filename)
 
    return files
