import logging
import sqlite3
import pandas as pd
from flyhostel.utils import get_dbfile
import numpy as np
logger=logging.getLogger(__name__)
from flyhostel.data.pose.landmarks import (
    distance_from_points_to_ellipse,
    distance_from_points_to_polygon,
)

class LandmarksLoader:
    basedir=None
    dt=None
    landmarks=None

    def load_landmarks(self):
        dbfile=get_dbfile(self.basedir)
        with sqlite3.connect(dbfile) as conn:
            self.landmarks=pd.read_sql(sql="SELECT * FROM LANDMARKS;", con=conn)
            roi_map=pd.read_sql(sql="SELECT * FROM ROI_MAP;", con=conn)

        roi_size=max(roi_map["w"].item(), roi_map["h"].item())
        landmarks_norm=[]
        for _, landmark in self.landmarks.iterrows():
            if landmark["shape"]=="food":
                landmark_norm=self.normalize_food_landmark(landmark.copy(), roi_size=roi_size)
            
            elif landmark["shape"]=="notch":
                landmark_norm=self.normalize_notch_landmark(landmark.copy(), roi_size=roi_size)
            
            else:
                raise ValueError(f"Landmark {landmark['shape']} not supported")
                
            
            landmarks_norm.append(landmark_norm)

        landmarks_norm=pd.concat(landmarks_norm, axis=0)
        self.landmarks["specification_norm"]=landmarks_norm["specification"].values

    @property
    def number_of_food_blobs(self):
        if self.landmarks is None:
            self.load_landmarks()
        food_blobs=self.landmarks.loc[self.landmarks["shape"]=="food"]
        return food_blobs.shape[0]

    def compute_if_fly_on_food_patch(self, include_outside=1):
        
        food_blobs=self.landmarks.loc[self.landmarks["shape"]=="food"]

        if food_blobs.shape[0]==0:
            logger.warning("No notch landmarks saved")
            return None
        
        self.dt["food_blobs"]=0
        across_blobs=[]

        j=0
        for _, food_blob in food_blobs.iterrows():
            ellipse=eval(food_blob["specification_norm"])
            in_ellipse_all=distance_from_points_to_ellipse(
                self.dt[["x", "y"]].values,
                ellipse["center"][0],
                ellipse["center"][1],
                ellipse["axes"][0]*include_outside,
                ellipse["axes"][1]*include_outside,
                np.radians(ellipse["angle"]),
            )
            self.dt[f"food_{j+1}_dist"]=in_ellipse_all
            # across_blobs.append(in_ellipse_all)
            j+=1
        
        # across_blobs=np.stack(across_blobs, axis=1)

        # check that one animal cannot be on two blobs at the same time
        # assert ((across_blobs>0).sum(axis=1)<=1).all()

        distances=self.dt[[f"food_{i+1}_dist" for i in range(self.number_of_food_blobs)]].values
        self.dt["food"]=distances.argmin(axis=1)+1
        self.dt["food_distance"]=distances.min(axis=1)


    def compute_if_fly_on_notch(self):
        
        notches=self.landmarks.loc[self.landmarks["shape"]=="notch"]

        if notches.shape[0]==0:
            logger.warning("No notch landmarks saved")
            return None
        
        j=0
        for _, notch in notches.iterrows():
            polygon=np.array(eval(notch["specification_norm"]))           
            assert polygon.shape==(4,2), f"{polygon.shape} != (4, 2)"
            points=self.dt[["x", "y"]].values
            self.dt[f"notch_{j+1}_dist"]=distance_from_points_to_polygon(points, polygon)
            j+=1

        distances=self.dt[[f"notch_{i+1}_dist" for i in range(notches.shape[0])]].values
        self.dt["notch"]=distances.argmin(axis=1)+1
        self.dt["notch_distance"]=distances.min(axis=1)


    @staticmethod
    def normalize_food_landmark(landmark, roi_size):
        data=eval(landmark["specification"])
        for feat in ["center", "axes"]:
            data[feat]=(np.array(data[feat])/roi_size).round(3).tolist()
        
        landmark["specification"]=str(data)
        return landmark
    
    
    @staticmethod
    def normalize_notch_landmark(landmark, roi_size):
        data=np.array(eval(landmark["specification"]))
        data/=roi_size
        data=data.round(3)
        landmark["specification"]=str(data.tolist())
        return landmark
