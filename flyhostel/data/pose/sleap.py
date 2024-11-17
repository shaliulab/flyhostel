import queue
import tempfile
import shutil
import glob
import itertools
import os.path
import logging
import sqlite3
import h5py

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import multiprocessing

logger=logging.getLogger(__name__)
from flyhostel.utils import get_dbfile
from flyhostel.data.pose.constants import chunksize as CHUNKSIZE
from flyhostel.data.pose.constants import bodyparts as BODYPARTS
from flyhostel.data.pose.constants import framerate as FRAMERATE
from ethoscopy.flyhostel import load_centroids
sleap_integration=logging.getLogger("sleap_integration")

import os
import subprocess

def concatenate_videos(video_list, output_path):
    # Create a temporary text file to list the videos
    with open('video_list.txt', 'w') as file:
        for video in video_list:
            file.write(f"file '{video}'\n")
    
    # Use ffmpeg to concatenate the videos
    ffmpeg_command = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', 'video_list.txt',
        '-c', 'copy', output_path
    ]
    
    subprocess.run(ffmpeg_command, check=True)
    
    # Clean up the temporary file
    os.remove('video_list.txt')



def add_text_to_frame(frame, text):

    # frame resolution = 1000x1000


    # Define the font, size, color, and thickness
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 0, 255)  # BGR (0, 0, 255) = red
    thickness = 2
    
    # Get the text size to position it properly
    text_size = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = int(frame.shape[1]*0.1)
    text_y = int(frame.shape[0]*0.1)

    
    # Add text to the frame
    frame_with_text = cv2.putText(
        img=frame,
        text=text,
        org=(text_x, text_y),
        fontFace=font,
        fontScale=font_scale,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )

    return frame_with_text

def annotate_data(input_video_path, output_video_path, df, column="text"):
    """
    Write text to an existing video

    input_video_path (str): Path to existing video
    output_video_path (str): Path to new video with annotation
    df (pd.DataFrame): One row per frame in the input video, with a column that contains what should be annotated
    """

    # Capture the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get the frame rate and size of the input video
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create a VideoWriter object to write the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if len(df) != frame_count:
        logger.warning(f"Error: The number of rows in the DataFrame does not match the number of frames in the video. ({len(df)}!={frame_count})")
    

    frame = 0
    pb=tqdm(total=frame_count, desc="Annotating")
    text=""
    while True:
        
        ret, img = cap.read()
        if not ret:
            break
        
        # Get the text for the current frame from the DataFrame
        try:
            text = df.loc[df["frame"]==frame, column].item()
        except:
            pass
            # text=""
        

        if not isinstance(text, str):
            import ipdb; ipdb.set_trace()
        # Add text to the frame
        height, width=img.shape[:2]
        img_with_text = cv2.resize(
            add_text_to_frame(
                cv2.resize(img, (1000, 1000)),
                text
            ),
            (width, height)
        )

        # Write the frame to the output video
        out.write(img_with_text)
        frame += 1
        pb.update(1)

    # Release the VideoCapture and VideoWriter objects
    cap.release()
    out.release()



try:
    from sleap.io.dataset import Labels
    from sleap.io.video import Video
    from sleap.instance import LabeledFrame, Instance, PredictedInstance, Track
    from sleap.io.visuals import resize_images, VideoMarkerThread
    from sleap.io.visuals import save_labeled_video
    cwd=os.getcwd()
    ref_labels_file="/Users/FlySleepLab_Dropbox/Data/flyhostel_data/fiftyone/FlyBehaviors/labels_train.slp"
    os.chdir(os.path.dirname(ref_labels_file))
    skeleton=Labels.load_file(ref_labels_file).skeleton
    os.chdir(cwd)

    
    def numpy(instance, bodyparts):
        
        """
        Arguments:

            instance (pd.Series): Coordinates of the nodes (bodyparts) of an instance
                in format node_x node_y (as columns of a pd.Series)
            bodyparts (list):
        Returns:
            np.array of shape (n_nodes, 2) of dtype float32
            containing the coordinates of the instance’s nodes.
            Missing/not visible nodes will be replaced with NaN.
        """    
        data=[]
        for bp in bodyparts:
            data.append(instance[[bp + "_x", bp + "_y"]].values.flatten())
        data=np.stack(data, axis=0)
        return data


    def make_labeled_frames(pose, frame_numbers, chunksize, video, identities, pb=True, predictions=False):

        labeled_frames=[]

        if pb:
            iterable=tqdm(frame_numbers)
        else:
            iterable=frame_numbers

        tracks={}
        for identity in identities:
            if identity is None:
                tracks[identity]=None
            else:
                tracks[identity]=Track(spawned_on=frame_numbers[0], name=f"Track-{identity}")


        for frame_number in iterable:
            frame_idx=frame_number%chunksize

            instance_series=pose.loc[(pose["frame_number"]==frame_number)]
            instances=[]
            for identity in identities:
                if identity is not None:
                    instance_series_identity=instance_series.loc[(instance_series["identity"]==identity)]
                else:
                    instance_series_identity=instance_series
                
                # if no data is found for the fly in that frame
                if instance_series_identity.shape[0]==0:
                    logger.warning("No data found for identity %s in frame %s", identity, frame_number)
                    continue
                base_instance_numpy=numpy(instance_series_identity, bodyparts=[node.name for node in skeleton.nodes])


                if predictions:
                    PredictedInstance.from_numpy( 
                        points=base_instance_numpy,
                        point_confidences=np.array([1.0,] * len(skeleton.nodes)),
                        instance_score=1.0,
                        skeleton=skeleton,
                        track=tracks[identity],
                    )
                else:
                    instance=Instance.from_numpy( 
                        points=base_instance_numpy,
                        skeleton=skeleton,
                        track=tracks[identity],
                    )
                instances.append(instance)

            lf = LabeledFrame(video=video, frame_idx=frame_idx, instances=instances)
            labeled_frames.append(lf)
        return labeled_frames


    def draw_frame(pose, index, identity, frame_number, chunksize=CHUNKSIZE, pb=True):
        """
        Just draw one frame, to be called by the user
        """
        frame_idx=frame_number % chunksize

        video=Video.from_filename(index.loc[index["frame_number"]==frame_number]["video"].item())
        labeled_frames=make_labeled_frames(
            pose,
            identities=[identity],
            frame_numbers=[frame_number],
            chunksize=chunksize, video=video,
            pb=pb
        )
        labels=Labels(labeled_frames=labeled_frames, videos=[video], skeletons=[skeleton])

        q1=queue.Queue()
        q2=queue.Queue()
        vmt=VideoMarkerThread(
            in_q=q1, out_q=q2, labels=labels, video_idx=0, scale=5,
            show_edges=True, edge_is_wedge=False, marker_size=1,
            crop_size_xy=False, color_manager=None, palette="standard", distinctly_color="instances"
        )

        loaded_chunk_idxs, video_frame_images=labels.videos[0].get_frames_safely([frame_idx])
        assert video_frame_images is not None
        assert len(loaded_chunk_idxs) > 0
        video_frame_images = resize_images(video_frame_images, 5)
        imgs=vmt._mark_images(loaded_chunk_idxs, video_frame_images)
        return imgs[0]


    def draw_video(pose, index, identities, frame_numbers, chunksize=CHUNKSIZE, output_filename=None, pb=True, **kwargs):
        """
        pose (pd.DataFrame): coordinates of body parts relative to the top left corner of a square around the fly.
            Needs to contain columns bp_x and bp_y for each bodypart, id (str), identity (int), frame_number.
        index (pd.DataFrame): timestamps and additional information relative to frames of a recording. Must contain columns frame_number and video.
        identity (int): Identity of the animal for which the video is to be made. Should match the identity of pose estimates stored in pose.
        frame_numbers (list): Frame numbers for which the video is to tbe made. Should match the frame_number of pose estimates stored in pose.
        """
        
        chunks=[frame_number // chunksize for frame_number in frame_numbers]
        assert len(set(chunks)) == 1, f"Please pass frames from within the same chunk"
        video_filename=index.loc[index["frame_number"]==frame_numbers[0], "video"].unique().item()
        video=Video.from_filename(video_filename)

        logger.debug("Making labeled frames")

        labeled_frames=make_labeled_frames(
            pose,
            identities=identities,
            frame_numbers=frame_numbers,
            chunksize=chunksize,
            video=video,
            pb=pb,
        )
        labels=Labels(labeled_frames=labeled_frames, videos=[video], skeletons=[skeleton])

        fn, extension = os.path.splitext(os.path.basename(video.backend.filename))

        if output_filename is None:
            output_filename=os.path.join(os.path.dirname(video.backend.filename), fn + "_render" + extension)

        Labels.save_file(labels, filename=output_filename+".slp")

        save_labeled_video(
            output_filename,
            labels=labels,
            video=labels.video,
            frames=frame_numbers%chunksize,
            # fps=fps,
            # scale=5.0,
            # crop_size_xy= None,
            # show_edges= True,
            # edge_is_wedge=False,
            # marker_size=1,
            # color_manager=None,
            # palette="standard",
            **kwargs
        )


    def draw_video_row(loader, identity, i, row, output, chunksize=45000, fps=15):
        index=loader.index_pandas[identity-1]
        os.makedirs(output, exist_ok=True)

        for dataset in ["raw", "jumps", "boxcar", "final"]:
            start = np.floor(row["start"]/loader.stride)*loader.stride
            end = np.ceil(row["end"]/loader.stride)*loader.stride
            
            if dataset=="raw":
                pose_dataset = loader.pose_raw
            elif dataset == "jumps":
                pose_dataset = loader.pose_jumps
            elif dataset == "boxcar":
                pose_dataset = loader.pose_boxcar
            elif dataset == "final":
                pose_dataset = loader.pose_interpolated

            folder=f"{output}/bout_{str(i).zfill(3)}"
            os.makedirs(folder, exist_ok=True)

            try:
                draw_video(
                    pose_dataset,
                    index,
                    identity=identity,
                    frame_numbers=np.arange(start, end, 10),
                    chunksize=chunksize,
                    fps=fps,
                    output_filename=f"{folder}/{dataset}.mp4"
                )
            except Exception as error:
                print(error)


    def get_local_identity(dbfile, chunk, identity):
        table_name="CONCATENATION_VAL"

        with sqlite3.connect(dbfile) as conn:
            cursor=conn.cursor()
            cmd=f"SELECT local_identity FROM {table_name} WHERE identity = {identity} AND chunk = {chunk};"
            print(cmd)
            cursor.execute(cmd)
            local_identity=int(cursor.fetchone()[0])
            return local_identity
            
    def make_pose_video_multi_fly(basedir, identities, frame_number, seconds, bodyparts, downsample, pose_name="raw", fps=None, prefix=None, palette="custom", data=None, single_animal=False, output_folder=".", extension=".mp4"):
        """
        Draw pose of an animal in the original video using a single job
        Please see NOTE in make_pose_video_multi_fly_mp if you update the signature 
        
        """
       
        frame_number1=int(frame_number+seconds*FRAMERATE)
        chunk=frame_number//CHUNKSIZE
        experiment="_".join(basedir.split('/')[-3:])
        number_of_animals=int(experiment.split("_")[1].replace("X",""))

        dbfile=get_dbfile(basedir)
        datasets=[]
        for identity in identities:
            if number_of_animals==1:
                local_identity=0
            else:
                local_identity=get_local_identity(dbfile, chunk, identity)
            if single_animal:
                video_file=f"{basedir}/flyhostel/single_animal/{str(local_identity).zfill(3)}/{str(chunk).zfill(6)}.mp4"
            else:
                video_file=f"{basedir}/{str(chunk).zfill(6)}.mp4"

            animal=f"{experiment}__{str(identity).zfill(2)}"
            
            #csv_file=f"{basedir}/flyhostel/single_animal/{str(local_identity).zfill(3)}/{str(chunk).zfill(6)}.csv"
            #if os.path.exists(csv_file):
            #    centroids=pd.read_csv(csv_file)[["frame_number", "x", "y"]]
            #else:
            #logger.debug("%s not found", csv_file)
            min_frame_number=chunk*CHUNKSIZE
            max_frame_number=(chunk+1)*CHUNKSIZE                
            centroids=load_centroids(dbfile, identity, min_frame_number, max_frame_number, roi_0_table="ROI_0_VAL", identity_table="IDENTITY_VAL")

            first_video=sorted(glob.glob(f"{basedir}/flyhostel/single_animal/{str(local_identity).zfill(3)}/*mp4"))[0]
            first_fn=int(os.path.splitext(os.path.basename(first_video))[0])*CHUNKSIZE
            f0=frame_number-first_fn
            f1=frame_number1-first_fn
            
            analysis_file=f"{basedir}/motionmapper/{str(identity).zfill(2)}/pose_{pose_name}/{animal}/{animal}.h5"
            with h5py.File(analysis_file, "r", locking=True) as f:
                pose=f["tracks"][0, :, :, f0:f1]
                bodyparts=[e.decode() for e in f["node_names"]]
                n_dims=pose.shape[0]

            dimensions=["x","y","z"][:n_dims]
            features_by_dimension={
                d: [bp + "_" + d for bp in bodyparts]
                for d in dimensions
            }
            bodyparts_nd=list(itertools.chain(*[[features_by_dimension[d][i] for d in dimensions] for i in range(len(bodyparts))]))
            pose_df=pd.DataFrame(pose.T.reshape((f1-f0, -1)))
            pose_df.columns=bodyparts_nd
            pose_df["frame_number"]=np.arange(frame_number, frame_number1)
            pose_df["frame_idx"]=pose_df["frame_number"]%CHUNKSIZE
            pose_df["identity"]=identity
            index=pose_df[["frame_number"]].copy()
            index["video"]=video_file
            index["identity"]=identity
            pose_df=pose_df.loc[pose_df["frame_number"]%downsample==0]
            pose_df=pose_df.merge(centroids, how="left", on="frame_number")
            
            for d in dimensions:
                assert pose_df[d].isna().mean()==0, f"Missing data for {d}"
                if not single_animal:
                    for feature in features_by_dimension[d]:
                        pose_df[feature]+=pose_df[d]-50

            datasets.append((pose_df, index))
        
        dataset=pd.concat([pose_df for pose_df, _ in datasets], axis=0)
        index=pd.concat([index for _, index in datasets], axis=0)
        if prefix is None:
            prefix=""
        elif not prefix.endswith("_"):
            prefix+="_"


        if output_folder is None:
            output_folder=tempfile.TemporaryDirectory().name
            print(f"Saving video to {output_folder}")

        
        if len(identities)==1:
            output_filename=os.path.join(output_folder, f"{prefix}{experiment}__{identities[0]}__{str(chunk).zfill(6)}_{frame_number}{extension}")
        else:
            output_filename=os.path.join(output_folder, f"{prefix}{experiment}__{str(chunk).zfill(6)}_{frame_number}{extension}")

        if fps is None:
            fps=max(int(FRAMERATE/downsample/3), 1)

        frame_numbers=np.array(sorted(np.unique(dataset["frame_number"].values)))
        draw_video(
            dataset, index, identities=identities,
            frame_numbers=frame_numbers,
            gui_progress=False, output_filename=output_filename, palette=palette,
            fps=fps, marker_size=.6,
            distinctly_color="instances",
            scale=4,
        )


        if data is not None:
            data=data[["id", "frame_number", "prediction", "text"]]
            # TODO: Support multiid
            data_index=pd.DataFrame({"frame_number": np.arange(
                data["frame_number"].iloc[0],
                data["frame_number"].iloc[-1]+1,
            )})
            data_full=data_index.merge(data, on=["frame_number"], how="left")
            data_full["prediction"].ffill(inplace=True)
            data_full["id"].ffill(inplace=True)
            data_full["text"].ffill(inplace=True)
            data_full["frame"]=np.arange(data_full.shape[0])
            output=output_filename.replace(".avi", "_annot.avi")
            annotate_data(output_filename, output, data_full, column="text")
            shutil.move(output, output_filename)

        return output_filename, dataset

    def worker(basedir, identities, fn0, block_seconds, *args, **kwargs):
        return make_pose_video_multi_fly(basedir, identities, fn0, block_seconds, *args, **kwargs)


    def make_pose_video_multi_fly_mp(basedir, identities, frame_number, seconds, n_jobs=1, block_seconds=1, data=None, output_folder=".", **kwargs):
        """
        Generate a video showing the pose predicted by the flyhostel pipeline
        """
        block_starts=[frame_number]
        block_frames=block_seconds*FRAMERATE
        
        
        os.makedirs(output_folder, exist_ok=True)
        

        n_blocks = int(np.ceil(seconds / block_seconds))
        for i in range(1, n_blocks):
            block_starts.append(
                int(block_starts[-1]+block_frames)
            
            )

        print(f"Blocks start at {block_starts}")

        if n_jobs is None:
            n_jobs = multiprocessing.cpu_count()  # or set your desired number of jobs
        pool = multiprocessing.Pool(processes=n_jobs)

        if data is None:
            datas=[None, ] * n_blocks
        else:
            datas=[]
            for i, _ in enumerate(block_starts):
                if (i+1) == len(block_starts):
                    datas.append(
                        data.loc[data["frame_number"]>=block_starts[i]]
                    )
                else:
                    datas.append(
                        data.loc[(data["frame_number"]>=block_starts[i]) & (data["frame_number"]<block_starts[i+1])]
                    )

        tasks = [
            # NOTE. The order of the arguments must match the signature of make_pose_video_multi_fly
            (
                # 0
                basedir,
                identities,
                fn0,
                block_seconds,
                kwargs.get("bodyparts", BODYPARTS),
                # 5
                kwargs.get("downsample", 1),
                kwargs.get("pose_name", "raw"),
                kwargs.get("fps", 1),
                kwargs.get("prefix", None),
                kwargs.get("palette", "solaris"),
                # 10
                datas[i],
                kwargs.get("single_animal", False),
                None,
            )
            for i, fn0 in enumerate(block_starts)
        ]
        tasks=[task for task in tasks if task[10] is not None and task[10].shape[0]>0]
        for task in tasks:
            if task[10] is not None:
                print(task[10].shape[0])

        if n_jobs==1:
            results=[worker(*tasks[0])]
        else:
            results = pool.starmap(worker, tasks)
            pool.close()
            pool.join()

        video_files=[e[0] for e in results]       
        output_video=os.path.join(output_folder, os.path.basename(video_files[0].replace(".avi", "_all.avi")))
        concatenate_videos(video_files, output_video)

        return output_video


except Exception as error:
    sleap_integration.debug("SLEAP cannot be loaded. SLEAP integration disabled")
    sleap_integration.debug(error)
    draw_video_row=None
    make_pose_video_multi_fly=None

