import re
import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import cupy as cp
import codetiming
import cmapy
from imgstore.interface import VideoCapture
from .pca import PCA
from flyhostel.utils.bbox import select_bounding_box
# from idtrackerai.trajectories import Trajectories
from monit.gpu import GPUMonitor

gpu_monitor = GPUMonitor(0)
gpu_monitor.start()

def get_gpu_usage():
    return gpu_monitor.get_stats()['memory']

class Preprocessing:
    
    
    def __init__(self, dimensions, median_body_length, *args, mode=1, **kwargs):

        self.mode = mode
        self._dimensions = dimensions
        self.median_body_length=median_body_length
        self.pca = PCA(n_components=dimensions, mode=mode)
        super(Preprocessing, self).__init__(*args, **kwargs)

    @staticmethod
    def compute_std_img(imgs):
        with cp.cuda.Device(0):
            with codetiming.Timer(text="{:.8f} seconds to send"):
                imgs_gpu = cp.asarray(imgs)
            with codetiming.Timer(text="{:.8f} seconds to compute"):
                std_gpu = cp.std(imgs_gpu, axis=0)
            with codetiming.Timer(text="{:.8f} seconds to get"):
                std_img = std_gpu.get()
                
        
        return std_img

    @staticmethod
    def compute_std_threshold(std_img):
        counts, values, hist = plt.hist(std_img.flatten(), bins=100)
        # std_threshold=values[np.argmax(np.diff(counts))]
        std_threshold = np.percentile(std_img.flatten(), 90)
        plt.axvline(std_threshold)
        plt.show()
        return std_threshold
    
    
    @staticmethod
    def compute_mask(std_img, threshold):
        significant_mask=np.uint8(255 * (std_img > threshold))
        print(f"{round(100 * significant_mask.mean() / 255, 2)} % of pixels kept")
        plt.imshow(significant_mask)
        plt.show()
        return significant_mask
    
    
    def _pca(self, data):
        print(f"Running PCA on {data.shape[0]} images")
        proj = self.pca.fit_transform(data.T)
        return self.pca, proj
    
    @staticmethod
    def make_eigenflies(pca, proj, mask):
        dimensions = proj.shape[1]
        resolution = mask.shape[:2][::-1]

        eigenflies = np.zeros((dimensions, *resolution), np.float64)
        placeholder=np.zeros((resolution[0] * resolution[1]), np.float64)
        for k in range(dimensions):
            eigenvector=pca.components_[:, k].copy()
            eigenfly=placeholder.copy()
            eigenfly[mask.flatten()==255]=eigenvector
            eigenflies[k]=cv2.normalize(eigenfly.reshape(resolution), eigenflies[k], 0, 255, cv2.NORM_MINMAX)
            assert eigenflies[k].max() > 0

        eigenflies=np.uint8(eigenflies)

        eigenflies_colored = []
        for eigenfly in eigenflies:
            eigenfly_colored = cv2.applyColorMap(eigenfly, cmapy.cmap('magma'))
            eigenfly_colored[mask==0]=255
            # print(eigenfly_colored.shape)
            eigenflies_colored.append(eigenfly_colored)
        eigenflies_colored=np.stack(eigenflies_colored)
        return eigenflies


    @staticmethod
    def plot_eigenflies(eigenflies):
        starts = np.arange(0, 50, 10)
        ends = starts+10
        rows=[]
        for i in range(len(starts)):
            rows.append(np.hstack(eigenflies[starts[i]:ends[i]]))

        collage=np.vstack(rows)
        collage.shape
        plt.rcParams["figure.figsize"]=(20, 10)
        plt.imshow(collage)
        

class DatasetManager(Preprocessing):
    """
    
    Call load_image_data with a dictionary of dataset: zt intervals
    
    Args:
        * videos_dirs (list): List of directories containing imgstores
        * segmented (bool): If True, imgstore should contain cropped and rotated single animals,
        otherwise, raw videos are expected and trajectories will be used to crop the animal in-situ
        TODO Actually implement the rotation in situ
    """
    
    def __init__(self, videos_dirs, *args, segmented=True, **kwargs):
        super(DatasetManager, self).__init__(*args, **kwargs)
        self.segmented = segmented
        self._videos_dirs = videos_dirs
        self._identities = {}
        if segmented:
            self.prefix = "fly_"
            self.load_identities()

        self.load_imgstores()           
        self.generate_datasetnames()
        
        self._h5_paths = [
            os.path.join(store._basedir, "proj.h5")
            for store in self._stores
        ]
        self._pca_paths = [
            os.path.join(store._basedir, "pca.npy")
            for store in self._stores
        ]

        self._extra_data = None
        self._projs_list = None
        self._masks = {}
        self._to_keep = [True for _ in self._datasetnames]
        
    @property
    def projs_list(self):
        return self._projs_list
    
    
    @property
    def datasetnames(self):
        return self._datasetnames

    
    def load_identities(self, identities=None):
        identities={}
        for vd in self._videos_dirs:
            folders_in_vd = os.listdir(vd)
            identities_in_vd = sorted([int(re.search(f"{self.prefix}(\d)", folder).group(1)) for folder in folders_in_vd])
            identities[vd]= identities_in_vd
            
        self._identities = identities

        
    def load_imgstores(self):
    
        stores = []
        for video_dir in self._videos_dirs:
            if self.segmented:
                for identity in self._identities[video_dir]:
                    stores.append(VideoCapture(os.path.join(video_dir, f"{self.prefix}{identity}", "metadata.yaml"), 2))
            else:
                stores.append(VideoCapture(os.path.join(video_dir, "metadata.yaml"), 2))

        for store in stores:
            assert not store.is_multistore
            
        self._stores = stores
        
        
    def generate_datasetnames(self):
        # get the name of the last three folders in the tree to build the name of the dataset
        self._datasetnames = ["-".join(store._basedir.split(os.path.sep)[-3:]) for store in self._stores]
        # get the hour part of the datetime YYYY-MM-DD_HH-MM-SS in the datasetname
        self._starttimes = [int(store._basedir.split(os.path.sep)[-3].split("_")[1].split("-")[0]) for store in self._stores]


    @staticmethod
    def skip_frame(extra_data, fn):
        if fn in extra_data.index:
            extra = extra_data.loc[fn]
            if len(extra["nearby_flies"]) > 0 or extra["blob_missing"]:
                skip=True
            else:
                skip=False
        else:
            skip=False
        
        return skip
        
        
    def load_chunks(self, store, idx, interval):

        
        zt0=store._metadata["zt0"] * 3600
        start_time = self._starttimes[idx] * 3600
        
        
        first_frame_time = interval[0].item() + zt0 - start_time
        last_frame_time =  interval[1].item() + zt0 - start_time
        
        start_chunk, _ = store._index.find_chunk_nearest(what="frame_time", value=first_frame_time*1000, past=True, future=False)
        end_chunk, _ = store._index.find_chunk_nearest(what="frame_time", value=last_frame_time*1000, past=False, future=True)

        assert start_chunk != -1
        assert end_chunk != -1
        chunks = list(range(start_chunk, end_chunk+1))
        return chunks
            
            
    def load_image_data(self, zt, mask=None):
        
        masks = []
        store_list = []
        frame_numbers_list = []
        extra_data_list = []
        
        
        for dataset in zt:

            idx = self._datasetnames.index(dataset)
            store=self._stores[idx]
            interval = np.array(zt[dataset]) * 3600
            chunks = self.load_chunks(store, idx, interval)
            extra_data=[]

            for chunk in chunks:
                df=store.get_extra_data(chunks=[chunk])
                if df is None:
                    import ipdb; ipdb.set_trace()
                df.set_index("frame_number", inplace=True)
                df["chunk"] = chunk
                extra_data.append(df)
            extra_data=pd.concat(extra_data)
            
            frame_numbers = []
            for chunk in chunks:
                frame_numbers.extend(
                    store._index.get_chunk_metadata(chunk)["frame_number"]
                )

            if mask is None:
                masks.append(self.compute_mask_from_imgs(store, frame_numbers, extra_data=extra_data))
                continue
                 
            else:
                self._masks[dataset] = mask
                store_list.append(store)
                frame_numbers_list.append(frame_numbers)
                extra_data_list.append(extra_data)
                trajectories_list = []
                for chunk in chunks:
                    trajectories_path = os.path.join(store._basedir, f"{str(chunk).zfill(6)}.npy")
                    print(f"Loading {trajectories_path}")
                    trajectory = np.load(trajectories_path)
                    trajectories_list.append(trajectory)
                pca, proj = self.run_pca(store_list, frame_numbers_list, trajectories_list, mask=mask, extra_data_list=extra_data_list)
                pd.DataFrame(proj).to_hdf(self._h5_paths[idx], "pca")
                np.save(self._pca_paths[idx], pca)     


        if mask is None:
            mask = np.stack(masks)
            thr=max(2, int(mask.shape[0] * 0.25))
            mask = np.uint8(255*(mask.sum(axis=0) > thr))
            return mask


    def run_pca(self, store_list, frame_numbers_list, trajectories_list, mask, extra_data_list):
        
        assert len(store_list) == len(frame_numbers_list)

        data_gpu = []
        if self.mode == 1:
            for i in range(len(store_list)):
                data_gpu.append(self.load_imgs(store=store_list[i], frame_numbers=frame_numbers_list[i], extra_data=extra_data_list[i], trajectories=trajectories_list[i], mask=mask))
            pca, proj = self._pca(data_gpu, mask)
  
        elif self.mode == 2:
            with cp.cuda.Device(0):
                for i in range(len(store_list)):
                    start = 0
                    chunk_size = 5000
                    end = min(start+chunk_size, len(frame_numbers_list[i]))
                    loop=1
                    total = math.ceil(len(frame_numbers_list[i]) / chunk_size)
                    while True:
                        fns = frame_numbers_list[i][start:end]
                        start+=chunk_size
                        if start >= len(frame_numbers_list[i]):
                            break

                        end = min(start+chunk_size, len(frame_numbers_list[i]))
                        data = self.load_imgs(store=store_list[i], frame_numbers=fns, extra_data=extra_data_list[i], mask=mask)
        
                        usage=get_gpu_usage()
                        print(f"GPU usage %: {usage}")                    

                        if usage < 90:
                            print(f"Sending {chunk_size} imgs from dataset {i} to gpu ({loop}/{total})")
                            data_gpu.append(cp.asarray(data))
                            print(f"GPU usage %: {get_gpu_usage()}")
                        else:
                            print(f"GPU is almost full. Not pushing new images") 
                        loop+=1

                data_gpu = cp.concatenate(data_gpu)
                pca, proj = self._pca(data_gpu)
                
                proj = proj.get()

        del data_gpu
        del self.pca.components_
        del self.pca.singular_values_
        del self.pca.cov_matrix
                
        return pca, proj


    def compute_mask_from_imgs(self, store, frame_numbers, extra_data):
        imgs = self.load_imgs(store, frame_numbers, extra_data)
        std_img = self.compute_std_img(imgs)
        threshold = self.compute_std_threshold(std_img)
        mask = self.compute_mask(std_img, threshold)
        return mask


    def load_imgs(self, store, frame_numbers, extra_data, trajectories=None, mask=None):
        imgs = []
        trajectory = []
        fn = frame_numbers[0]
        pb=tqdm(total=len(frame_numbers))
        img, meta = store.get_data(frame_numbers[0])

        while fn < frame_numbers[-1]:
            if self.skip_frame(extra_data, fn):
                pass
                # print(f"Skipping {fn}")
            else:
                img_flat = img.flatten()
                if mask is not None:
                    data = img_flat[mask.flatten() == 255]
                else:
                    data = img_flat

                imgs.append(data)
            centroids = meta[2]
            trajectory.append(centroids)
            fn = store.frame_number
            img, meta = store.get_data(fn)
            if trajectories is not None:
                img = select_bounding_box(img, trajectories[fn], self.median_body_length)
            pb.update(1)
        trajectory=np.vstack(trajectory)
        imgs=np.stack(imgs)
        return imgs


    def load_decomposition(self,datasets):

        self._projs_list = []
        
        for dataset in datasets:
            try:
                idx = self._datasetnames.index(dataset)
            except ValueError:
                print(f"{dataset} not available")
                continue
            
            h5 = self._h5_paths[idx]
            
        
            if os.path.exists(h5):
                print(f"Loading {h5}")
                df = pd.read_hdf(h5)
                self._projs_list.append(df.iloc[:, :self._dimensions])
            else:
                print(f"{h5} does not exist")
                self._to_keep[idx]=False

        self._datasetnames = [v for idx, v in enumerate(self._datasetnames) if self._to_keep[idx]]
        self._stores = [v for idx, v in enumerate(self._stores) if self._to_keep[idx]]
        self._pca_paths = [v for idx, v in enumerate(self._pca_paths) if self._to_keep[idx]]
        self._h5_paths = [v for idx, v in enumerate(self._h5_paths) if self._to_keep[idx]]
        self._to_keep = [True for _ in self._datasetnames] 
