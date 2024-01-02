import queue
import os.path
import logging
import numpy as np
from tqdm.auto import tqdm

logger=logging.getLogger(__name__)

try:
    from sleap.io.dataset import Labels
    from sleap.io.video import Video
    from sleap.instance import LabeledFrame, Instance
    from sleap.io.visuals import resize_images, VideoMarkerThread
    from sleap.io.visuals import save_labeled_video
    cwd=os.getcwd()
    ref_labels_file="/Users/FlySleepLab_Dropbox/Data/flyhostel_data/fiftyone/FlyBehaviors/FlyBehaviors_6cm.v003.slp"
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


    def make_labeled_frames(pose, identity, frame_numbers, chunksize, video):

        labeled_frames=[]

        for frame_number in tqdm(frame_numbers):
            frame_idx=frame_number%chunksize

            instance_series=pose.loc[(pose["identity"]==identity) & (pose["frame_number"]==frame_number)]
            base_instance_numpy=numpy(instance_series, bodyparts=[node.name for node in skeleton.nodes])


            instance=Instance.from_numpy( 
                base_instance_numpy, skeleton=skeleton
            )
            lf = LabeledFrame(video=video, frame_idx=frame_idx, instances=[instance])
            labeled_frames.append(lf)
        return labeled_frames


    def draw_frame(pose, index, identity, frame_number, chunksize=45000):
        frame_idx=frame_number % chunksize

        video=Video.from_filename(index.loc[index["frame_number"]==frame_number]["video"].item())
        labeled_frames=make_labeled_frames(
            pose, identity,
            frame_numbers=[frame_number], chunksize=chunksize, video=video
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


    def draw_video(pose, index, identity, frame_numbers, chunksize=45000, fps=15, output_filename=None):
        """
        pose (pd.DataFrame): coordinates of body parts relative to the top left corner of a square around the fly.
            Needs to contain columns bp_x and bp_y for each bodypart, id (str), identity (int), frame_number.
        index (pd.DataFrame): timestamps and additional information relative to frames of a recording. Must contain columns frame_number and video.
        identity (int): Identity of the animal for which the video is to be made. Should match the identity of pose estimates stored in pose.
        frame_numbers (list): Frame numbers for which the video is to tbe made. Should match the frame_number of pose estimates stored in pose.
        """
        
        chunks=[frame_number // chunksize for frame_number in frame_numbers]
        assert len(set(chunks)) == 1, f"Please pass frames from within the same chunk"

        video=Video.from_filename(index.loc[index["frame_number"]==frame_numbers[0], "video"].item())


        logger.debug("Making labeled frames")

        labeled_frames=make_labeled_frames(
            pose, identity,
            frame_numbers=frame_numbers, chunksize=chunksize, video=video
        )
        labels=Labels(labeled_frames=labeled_frames, videos=[video], skeletons=[skeleton])

        fn, extension = os.path.splitext(os.path.basename(video.backend.filename))   

        if output_filename is None:
            output_filename=os.path.join(os.path.dirname(video.backend.filename), fn + "_render" + extension)

        save_labeled_video(
            output_filename,
            labels=labels,
            video=labels.video,
            frames=frame_numbers%chunksize,
            fps=fps,
            scale=5.0,
            crop_size_xy= None,
            show_edges= True,
            edge_is_wedge=False,
            marker_size=1,
            color_manager=None,
            palette="standard",
            distinctly_color="instances",
            gui_progress=False
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

except Exception as error:
    print("SLEAP cannot be loaded. SLEAP integration disabled")
    draw_video_row=None
    print(error)