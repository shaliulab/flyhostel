from abc import ABC, abstractmethod
import os.path
import hdf5storage
import numpy as np

import pandas as pd
from tqdm.auto import tqdm

from flyhostel.data.pose.constants import chunksize

NUMBER_OF_SAMPLES={"walk": 30_000, "inactive": 10_000, "groom": 30_000}

LTA_DATA=os.environ["LTA_DATA"]

class WaveletLoader(ABC):

    datasetnames=[]
    identities=[]
    experiment=None
    pose_boxcar=None
    deg=None
    
    @abstractmethod
    def annotate_pose(pose, behaviors):
        raise NotImplementedError()
    

    def load_wavelets(self, identity=None, matfile=None):
        """
        Load pre-computed wavelet transform of pose
        """
        wavelets=[]
        frequencies=None
        freq_names=None

        previous_freq_names=None
        if matfile is not None:
            assert len(self.datasetnames)==1

        if identity is None:
            datasetnames=self.datasetnames
        else:
            datasetnames=[self.datasetnames[identity-1]]


        for i, datasetname in enumerate(tqdm(datasetnames, desc='loading wavelet dataset')):
            if matfile is None:
                matfile=os.path.join(LTA_DATA, "Wavelets", datasetname + "-pcaModes-wavelets.mat")

            if not os.path.exists(matfile):
                print(f"{matfile} not found")
                continue

            data=hdf5storage.loadmat(matfile)
            freq_names=[f.decode() for f in data["freq_names"]]
            wavelets_single_animal=pd.DataFrame(data["wavelets"], columns=freq_names)
            if previous_freq_names is None:
                previous_freq_names=freq_names
            assert all([a == b for a, b in zip(freq_names, previous_freq_names)])
            frequencies=data["f"]
            
            if identity is None:
                id=self.identities[i]
            else:
                id=self.identities[identity-1]

            wavelets_single_animal["id"]=id

            wavelets_single_animal.set_index("id", inplace=True)
            wavelets.append(wavelets_single_animal)
        
        if len(datasetnames)>1:
            import ipdb;ipdb.set_trace()

        wavelets=pd.concat(wavelets, axis=0)
        return wavelets, (frequencies, freq_names)



    def load_dataset(self, pose=None, wavelets=None):
        """"
        Generate dataset of pose + wavelets 

        Wavelets are assumed to be pre-computed and stored in LTA_DATA/Wavelets/datasetname-pcaModes-wavelets.mat
        """

        # load pose data and annotate it using ground truth
        #####################################################
        if pose is None:
            pose=self.pose_boxcar.copy()

        pose["frame_idx"]=pose["frame_number"]%chunksize
        pose["chunk"]=pose["frame_number"]//chunksize
        
        pose_annotated=self.annotate_pose(pose, self.deg)

        # load wavelet transform of the data
        #####################################################
        out=self.load_wavelets(matfile=wavelets)
        if out is None:
            raise Exception(f"Wavelets of experiment {self.experiment} cannot be loaded. Did you generate them?")
        else:
            wavelets, (frequencies, freq_names)=out
    
        
        # merge pose and wavelet information
        # NOTE This assaumes the frame number of the wavelet is the same as the pose
        wavelets["frame_number"]=pose_annotated["frame_number"].values
        pose_annotated.set_index(["id", "frame_number"], inplace=True)
        wavelets.set_index("frame_number", append=True, inplace=True)
        
        # wavelets=wavelets.reset_index().set_index(["id", "frame_number"])
        pose_annotated_with_wavelets=pd.merge(pose_annotated, wavelets, left_index=True, right_index=True)
        del wavelets
        pose_annotated_with_wavelets.reset_index(inplace=True)

        # generate a dataset of wavelets and the ground truth for all behaviors
        ##########################################################################
        pe_inactive=pose_annotated_with_wavelets.loc[pose_annotated_with_wavelets["behavior"]=="pe_inactive"]
        behaviors=np.unique(pose_annotated_with_wavelets["behavior"]).tolist()
        for behav in ["unknown", "pe_inactive"]:
            if behav in behaviors:
                behaviors.pop(behaviors.index(behav))


        dfs=[pe_inactive]
        for behav in behaviors:
            d=pose_annotated_with_wavelets.loc[pose_annotated_with_wavelets["behavior"]==behav].sample(frac=1).reset_index(drop=True)
            samples_available=d.shape[0]
            if behav=="pe_inactive":
                n_max=samples_available
            else:
                max_seconds=60
                n_max=6*max_seconds
            number_of_samples=NUMBER_OF_SAMPLES.get(behav, n_max)
            dfs.append(
                d.iloc[:number_of_samples]
            )

        labeled_dataset = pd.concat(dfs, axis=0)
        unknown_dataset = pose_annotated_with_wavelets.loc[pose_annotated_with_wavelets["behavior"]=="unknown"]

        return labeled_dataset, unknown_dataset, (frequencies, freq_names)
