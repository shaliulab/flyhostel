import os.path
import matplotlib.pyplot as plt
plt.set_cmap("gray")
plt.rcParams["figure.figsize"] = (3, 3)
import pandas as pd
import numpy as np
from imgstore.interface import VideoCapture

from idtrackerai.postprocessing.individual_videos import (
    generate_individual_video_rotation_invariant,
    compute_width_height_individual_video,
)
from idtrackerai.video import Video
from feed_integration.lists_of_blobs import CachedListOfBlobs
from feed_integration.idtrackerai.paths import load_number_of_animals

from .parser import get_parser

def main(args=None, ap = None):
    if args is None:
        ap = get_parser(ap)
        args = ap.parse_args()

    chunks = list(range(*args.interval))

    store_path=args.store_path

    CHUNK=chunks[0]

    store = VideoCapture(store_path, CHUNK)

    assert not store.is_multistore
    number_of_animals=load_number_of_animals(store)

    lists_of_blobs = CachedListOfBlobs(
        store, chunk=CHUNK, overlap=True,
        load_from_data=True, n_jobs=1,
        allow_all_collections=None,
        can_load_from_cache=True,
    )

    cwd = os.getcwd()
    try:
        os.chdir(
            os.path.join(
                store.get_root(),
                "idtrackerai"
            )
        )
        video = Video(video_path=store_path, open_multiple_files=False, chunk=chunks[0])
        basedir = store._basedir.rstrip(os.path.sep)
        experiment_name = os.path.basename(basedir)
        video._session_folder = basedir
        lists_of_blobs[chunks[0]]
        ref_video = lists_of_blobs._videos[chunks[0]]
        video._user_defined_parameters=ref_video._user_defined_parameters.copy()
        video._user_defined_parameters["number_of_animals"]=1
        video.chunksize = store._metadata["chunksize"]
        video._median_body_length = ref_video._median_body_length

        trajectories_path=os.path.join(store.get_root(), "flyhostel", f"{experiment_name}_trajectories.npy")
        concatenation_path=os.path.join(store.get_root(), "flyhostel", f"{experiment_name}_concatenation.csv")
        concatenation=np.loadtxt(concatenation_path, np.int32)
        video.concatenation=concatenation

        trajectories = np.load(trajectories_path, allow_pickle=True)
        
        video.create_individual_videos_folder()
        
        assert video.individual_videos_folder
        assert video.chunksize
        assert video.frames_per_second
        assert video.median_body_length_full_resolution
        
        height, width = compute_width_height_individual_video(video)

        for identity in range(1, number_of_animals+1):
            generate_individual_video_rotation_invariant(
                video, trajectories, lists_of_blobs,
                identity=identity, width=width, height=height
            )


    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    main()