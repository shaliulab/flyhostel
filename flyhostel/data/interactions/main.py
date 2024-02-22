  
from flyhostel.data.pose.main import FlyHostelLoader
from flyhostel.data.groups.group import FlyHostelGroup
from flyhostel.data.pose.constants import bodyparts as BODYPARTS
from flyhostel.data.pose.constants import SQUARE_WIDTH, SQUARE_HEIGHT
from flyhostel.data.pose.constants import framerate as FRAMERATE



def compute_experiment_interactions(experiment, number_of_animals, output=None, dist_max_mm=2.5, min_interaction_duration=.3):
    loaders=[
        FlyHostelLoader(
            experiment=experiment,
            identity=identity,
            chunks=range(0, 400),
            identity_table="IDENTITY_VAL",
            roi_0_table="ROI_0_VAL"
        )
        for identity in range(1, number_of_animals+1)
    ]

    # from centroid data
    ################################
    group=FlyHostelGroup.from_list(loaders, protocol="centroids", dist_max_mm=dist_max_mm, min_interaction_duration=min_interaction_duration)
    dt=group.load_centroid_data()
    # assume the thorax is where the centroid is,
    # which is the middle of the frame
    dt["thorax_x"]=SQUARE_WIDTH//2 
    dt["thorax_y"]=SQUARE_HEIGHT//2

    pose=dt[["id", "identity", "frame_number", "thorax_x", "thorax_y"]],
    dt=dt[["id", "identity", "frame_number", "x", "y"]],
    bodyparts=["thorax"]
    ################################

    # from pose data
    ################################
    group=FlyHostelGroup.from_list(loaders, protocol="full", dist_max_mm=dist_max_mm, min_interaction_duration=min_interaction_duration)
    dt=group.load_centroid_data()
    pose=group.load_pose_data("pose_boxcar")
    ################################


    # finally
    interactions = group.find_interactions(
        dt,
        pose,
        bodyparts=BODYPARTS,
        framerate=FRAMERATE
    )

    if output is not None:
        interactions.to_csv(output)
    return interactions