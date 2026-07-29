class GenericLoader:

    def __init__(self, experiment: str, identity: int, pixels_per_mm: float, chunksize: int, framerate: int, square_width: int, **kwargs):
        self.experiment=experiment              # shared by all flies in the group
        self.identity=identity                  # specific to this fly in this group
        self.pixels_per_mm=pixels_per_mm        # how many pixels make up one mm in the arena?
        self.chunksize=chunksize                # how many frames make up one of the videos that make up the recording
        self.framerate=framerate                # how many frames are collected in one second?
        self.square_width=square_width          # how many pixels does the square around the centroid have
        self.square_height=square_height
        self.kwargs=kwargs



    @property
    def dt(self):
        # implement here the logic to return a pd.DataFrame that contains for every frame in the pose
        # frame_number
        # center_x                     (in raw pixels coordinates)
        # center_y                     (in raw pixels coordinates)
        raise NotImplementedError


    def get_pose_file_h5py(self, pose_name: str ="raw"):
        # implement here the logic that will get the path to the .h5 of this file
        # you can use
        # self.experiment
        # self.identity
        # self.kwargs (you can pass here whatever you want!)
        raise NotImplementedError

    