import os.path
import cv2

class ImageWriter:
    def __init__(self, path, codec, fps, shape, isColor, counter=0, overwrite=False):
        self._path=os.path.splitext(path)[0]
        self.counter=counter
        os.makedirs(self._path, exist_ok=overwrite)

    def write(self, frame):
        file_path=os.path.join(self._path, f"img_{str(self.counter).zfill(10)}.png")
        cv2.imwrite(file_path, frame)
        self.counter+=1
        
    def release(self):
        self.counter=0
        pass