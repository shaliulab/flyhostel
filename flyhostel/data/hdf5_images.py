import warnings

import h5py
import cv2
import numpy as np
from tqdm.auto import tqdm

class HDF5ImagesReader:
    
    
    def __init__(self, hdf5_files, width, height, resolution, background_color, chunk):
        
        self._files = hdf5_files
        self._file_idx = -1
        self._key_counter = 0
        self._file = None
        #self._tqdm=tqdm(total=len(self._files), desc=f"Processing chunk {chunk}")
        self._tqdm=None
        self._update_filehandler()
        self._finished = False
        self.width = width
        self.height = height
        self.background_color = background_color
        self.resolution = resolution
        self._NULL = np.ones((self.height, self.width), np.uint8) * self.background_color
        self._chunk = chunk
        print(f"resolution of image reader {self.resolution}")

    @property
    def key(self):
        try:
            v=self._keys[self._key_counter]
            return v
        except IndexError as error:
            print(f"{self._key_counter} is not found in self._keys of length {len(self._keys)}")
            raise error
        
    @property
    def frame_number(self):
        return int(self.key.split("-")[0])
    
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        
        
    def close(self):
        pass

    
    @staticmethod
    def edit_image(img, width, height, background_color):
        img=cv2.copyMakeBorder(img, 0, max(0, height-img.shape[0]), 0, max(0, width-img.shape[1]), cv2.BORDER_CONSTANT, None, background_color)

        if img.shape[0] > height:
            #logger.debug(f"Chunk {chunk} - frame_number {frame_number}. Cropping {img.shape[0]-height} pixels along Y dim")
            top = (img.shape[0] // 2 - height // 2)
            img=img[top:(top+height), :]
        if img.shape[1] > width:
            #logger.debug(f"Chunk {chunk} - frame_number {frame_number}. Cropping {img.shape[1]-width} pixels along X dim")
            left = (img.shape[1] // 2 - width // 2)
            img=img[:, left:(left+width)]

        img=cv2.copyMakeBorder(img, 0, max(0, height-img.shape[0]), 0, max(0, width-img.shape[1]), cv2.BORDER_CONSTANT, None, background_color)
        assert img.shape[0] == height, f"{img.shape[0]} != {height}"
        assert img.shape[1] == width, f"{img.shape[1]} != {width}"
        return img

    
    @staticmethod
    def _resize_to_resolution(img, resolution):
        if img.shape != resolution[::-1]:
            img = cv2.resize(img, resolution[::-1], cv2.INTER_AREA)
        return img


    def _read(self):
        img_ = self._file[self.key][:]
        img_ = self.edit_image(img_, self.width, self.height, self.background_color)
        img_ = self._resize_to_resolution(img_, self.resolution)
        #if img_.shape[0]>200: import ipdb; ipdb.set_trace()
        self._key_counter+=1
        self._check_end_of_file() 
        return img_


    def _fetch(self, frame_number, in_frame_index):
         modified_key = f"{frame_number}-{in_frame_index}-modified"
         # NOTE
         # This assumes the modified version of the key
         # will exist in the next position
         try:
             if self._key_counter < (len(self._keys)-1) and self._keys[self._key_counter+1] == modified_key:
                 self._key_counter+=1
         except Exception as error:
             print(self._key_counter, len(self._keys))
             print(error)
         
         return self._read()

    def read(self, frame_number, number_of_animals):

        if self._finished:
            return None

        arrs = []
        for i in range(number_of_animals):

            frame_number_ = self.frame_number
            if frame_number is None:
                frame_number=frame_number_
                arrs.append(self._fetch(frame_number, i))
            else:
                # the frame number requested is less than the current key
                # which means there is no DATA and we should add NULL
                if frame_number < frame_number_:
                    arrs.append(self._NULL.copy())
                # the frame number requested
                # is the same as the current key
                elif frame_number == frame_number_:
                    arrs.append(self._fetch(frame_number, i))
                # the frame number requested 
                # is more than the current key
                # which means there are too many entries in a frame
                # and we need to move forward until they agree again
                else:
                    import ipdb; ipdb.set_trace()
                    count=0
                    while frame_number > frame_number_:
                        self._key_counter+=1
                        self._check_end_of_file() 
                        frame_number_ = self.frame_number
                        count+=1

                    warnings.warn(f"Skipped {count} keys to deliver frame number {frame_number}")
                    arrs.append(self._read())
       
        assert len(arrs) == number_of_animals
        img = np.hstack(arrs)

        return frame_number, img


    def _check_end_of_file(self):
        
        if len(self._keys) == self._key_counter:
            if (self._file_idx+1) == len(self._files):
                self._finished=True
                return None
            else:
                self._update_filehandler()

    def _update_filehandler(self):
        self._file_idx +=1
        if self._file is not None:
            self._file.close()
        
        if self._tqdm is not None: self._tqdm.update(1)
        self._file = h5py.File(self._files[self._file_idx], "r")
        self._keys = list(self._file.keys())
        self._key_counter=0

