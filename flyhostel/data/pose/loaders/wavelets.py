from abc import ABC, abstractmethod
import logging
import os.path
import h5py
import numpy as np

import pandas as pd
from tqdm.auto import tqdm
from flyhostel.utils import get_chunksize, get_framerate
logger=logging.getLogger(__name__)

class WaveletLoader:

    def __init__(self, *args, **kwargs):
        self.datasetnames=[]
        self.identities=[]
        self.identity=None
        self.basedir=None
        self.experiment=None
        self.pose_boxcar=None
        self.deg=None
        self.dt=None
        self.wavelets=None
        self.ids=[]
        super(WaveletLoader, self).__init__(*args, **kwargs)

    @abstractmethod
    def annotate_pose(self, pose, behaviors):
        raise NotImplementedError()


    def get_matfile(self):
        animal=self.experiment + "__" + str(self.identity).zfill(2)
        return os.path.join(
            self.basedir, "motionmapper", str(self.identity).zfill(2),
            "wavelets/FlyHostel_long_timescale_analysis/Wavelets",
            animal + "-pcaModes-wavelets.mat"
        )


    def load_wavelets(self, matfile=None, frames=None):
        """
        Load pre-computed wavelet transform of pose

        frames (list): List of frames whose wavelet is to be loaded.
          Relative to the start of the wavelets dataset
          i.e. not necessarily the same start as the recording because
          1) start of wavelets may not be the same as the recording
          2) not all original frames may have a wavelet transform

        """
        # wavelets=[]
        frequencies=None
        freq_names=None
        previous_freq_names=None

        if matfile is None:
            matfile=self.get_matfile()

        if not os.path.exists(matfile):
            logger.error("%s not found", matfile)
            return None, (None, None)

        logger.debug("Loading %s", matfile)
        with h5py.File(matfile, "r") as data:
            freq_names=[f.decode() for f in data["freq_names"][:]]

            if frames is None:
                logger.debug("Loading wavelets in all frames")
                wavelets_single_animal=pd.DataFrame(data["wavelets"][:], columns=freq_names)
            else:
                logger.debug("Loading wavelets in %s frames", len(frames))
                wavelets_single_animal=pd.DataFrame(data["wavelets"][frames, ...], columns=freq_names)


            if previous_freq_names is None:
                previous_freq_names=freq_names
            assert all([a == b for a, b in zip(freq_names, previous_freq_names)])
            frequencies=data["f"][:]


        wavelets_single_animal["id"]=self.ids[0]

        wavelets_single_animal.set_index("id", inplace=True)
        # wavelets.append(wavelets_single_animal)
        wavelets=wavelets_single_animal
        # wavelets=pd.concat(wavelets, axis=0)
        return wavelets, (frequencies, freq_names)



    def load_dataset(self, feature_types=None, pose=None, wavelets=None, segregate=True, deg=None):
        """"
        Generate dataset of pose + wavelets + centroid position

        Wavelets are assumed to be pre-computed and stored in motionmapper/identity/wavelets/FlyHostel_long_timescale_analysis/Wavelets/datasetname-pcaModes-wavelets.mat
        """

        # load pose data and annotate it using ground truth
        #####################################################
        if deg is None:
            deg=self.deg

        if pose is None:
            pose=self.pose_boxcar.copy()
        else:
            pose=pose.copy()

        chunksize=get_chunksize(self.experiment)
        framerate=get_framerate(self.experiment)
        
        if feature_types is None or "centroid_speed" in feature_types:
            dt=self.dt[["frame_number", "x", "y"]]
            dt=dt.loc[(dt["frame_number"]%framerate)==0]
            dt["diff_x"]=[0]+dt["x"].diff().values[1:].tolist()
            dt["diff_y"]=[0]+dt["y"].diff().values[1:].tolist()
            dt["centroid_speed"]=np.sqrt((dt["diff_x"]**2+dt["diff_y"]**2))
            pose["frame_number_round"]=framerate*(pose["frame_number"]//framerate)
            pose=pose.merge(dt[["frame_number", "centroid_speed"]].rename({"frame_number": "frame_number_round"}, axis=1), how="left", on="frame_number_round")
            pose.sort_values("frame_number", inplace=True)
            pose["centroid_speed"].ffill(inplace=True)
            pose.drop("frame_number_round", axis=1)

        pose=self.annotate_pose(pose, deg)
        pose["frame_idx"]=pose["frame_number"]%chunksize
        pose["chunk"]=pose["frame_number"]//chunksize


        # load wavelet transform of the data
        #####################################################
        out=self.load_wavelets(matfile=wavelets)
        if out is None:
            raise Exception(f"Wavelets of experiment {self.experiment} cannot be loaded. Did you generate them?")
        else:
            wavelets, (frequencies, freq_names)=out
            
        features=freq_names

        # merge pose and wavelet information
        # NOTE This assumes the frame number of the wavelet is the same as the pose
        assert wavelets.shape[0]==pose.shape[0], f"Wavelets has {wavelets.shape[0]} rows, but pose has {pose.shape[0]} rows. They should be the same"
        wavelets["frame_number"]=pose["frame_number"].values

        pose.set_index(["id", "frame_number"], inplace=True)
        wavelets.set_index("frame_number", append=True, inplace=True)

        pose_with_wavelets=pd.merge(pose, wavelets, left_index=True, right_index=True)
        del wavelets
    
        if feature_types is None or "centroid_speed" in feature_types:
            features+=["centroid_speed"]
        pose_with_wavelets.reset_index(inplace=True)


        if segregate:
            labeled_dataset = pose_with_wavelets.loc[pose_with_wavelets["behavior"]!="unknown"]
            unknown_dataset = pose_with_wavelets.loc[pose_with_wavelets["behavior"]=="unknown"]
            return labeled_dataset, unknown_dataset, (frequencies, freq_names, features)
        else:
            return pose_with_wavelets, (frequencies, freq_names, features)


