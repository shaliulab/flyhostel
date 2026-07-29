import pandas as pd

def get_basedir(experiment, identity):
    raise NotImplementedError

class GenericLoader:

    def __init__(self, experiment: str, identity: int, **kwargs):
        self.experiment=experiment                  # shared by all flies in the group
        self.identity=identity                      # specific to this fly in this group

        # TODO This is probably constant in all experiments, so you can just hardcode the value here
        # self.pixels_per_mm=                # how many pixels make up one mm in the arena?
        # self.chunksize=                    # how many frames make up one of the videos that make up the recording
        # self.framerate=                    # how many frames are collected in one second?
        # self.square_width=                 # how many pixels does the square around the centroid have
        # self.square_height=
        # self.roi_size =                    # shortest length of the raw frame, can be left as None if no landmarks are present
        self.basedir=get_basedir(experiment, identity)
        self.landmarks=pd.DataFrame({"shape": []})
        self.kwargs=kwargs
        self.dt=None


    @property
    def datasetnames(self):
        return [f"{self.experiment}__{self.identity}"]


    def load_centroid_data(self, cache):
        # implement here the logic to populate self.dt with a pd.DataFrame that contains for every frame in the pose
        # frame_number
        # center_x                     (in raw pixels coordinates)
        # center_y                     (in raw pixels coordinates)
        self.dt=None
        return 

         

    def get_pose_file_h5py(self, pose_name: str ="raw"):
        # implement here the logic that will get the path to the .h5 of this file
        # you can use
        # self.experiment
        # self.identity
        # self.kwargs (you can pass here whatever you want!)
        raise NotImplementedError

    