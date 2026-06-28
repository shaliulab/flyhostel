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
    landmarks_all=None
    

    def load_landmarks(self, errors="ignore"):
        dbfile=get_dbfile(self.basedir)
        with sqlite3.connect(dbfile) as conn:
            self.landmarks=pd.read_sql(sql="SELECT * FROM LANDMARKS;", con=conn)
            roi_map=pd.read_sql(sql="SELECT * FROM ROI_MAP;", con=conn)

        roi_size=max(roi_map["w"].item(), roi_map["h"].item())
        index=[]
        specification=[]
        for idx, landmark in self.landmarks.iterrows():
            if landmark["shape"]=="food":
                landmark_norm=self.normalize_food_landmark(landmark.copy(), roi_size=roi_size)
                specification.append(landmark_norm["specification"])
                index.append(idx)

            elif landmark["shape"]=="notch":
                landmark_norm=self.normalize_notch_landmark(landmark.copy(), roi_size=roi_size)
                specification.append(landmark_norm["specification"])
                index.append(idx)

            elif landmark["shape"]=="Polygon":
                logger.warning(f"Ignoring landmark {landmark}")
 
            elif errors=="raise":
                raise ValueError(f"Landmark {landmark['shape']} not supported")
            elif errors=="ignore":
                logger.warning("Landmark %s not supported", landmark['shape'])
            
        self.landmarks_all=self.landmarks.copy()
        self.landmarks=self.landmarks.loc[index]
        self.landmarks["specification_norm"]=specification

    def load_landmarks_and_compute_distances(self):
        assert self.dt is not None
        self.load_landmarks(errors="raise")
        self.compute_if_fly_on_food_patch(include_outside=1)
        self.compute_if_fly_on_notch()
        self.compute_distance_to_edge()


    def compute_distance_to_edge(self):
        """
        Compute distance from each fly position to the arena edge.
        Uses the arena mask (ROI_MAP) stored in the database.
        """
        import sqlite3
        from scipy import ndimage
        from PIL import Image
        import io
        
        # Create SQLite connection from file path
        conn = sqlite3.connect(self.dbfile)
        
        try:
            # Load arena mask from database
            roi_map = pd.read_sql("SELECT mask FROM ROI_MAP LIMIT 1", conn)
        except Exception as e:
            logger.warning(f"Could not read ROI_MAP: {e}")
            conn.close()
            return None
        
        if roi_map.empty:
            logger.warning("No ROI_MAP found in database")
            conn.close()
            return None
        
        conn.close()
        
        # Deserialize the BLOB (binary mask image)
        mask_blob = roi_map.iloc[0]["mask"]
        
        # Mask is stored as binary image data
        try:
            # If stored as raw numpy array bytes
            mask = np.frombuffer(mask_blob, dtype=np.uint8).reshape(-1, -1)
        except:
            try:
                # If stored as PIL Image BLOB
                mask = np.array(Image.open(io.BytesIO(mask_blob)))
            except Exception as e:
                logger.warning(f"Could not deserialize mask blob: {e}")
                return None
        
        # Ensure binary (white=1, black=0)
        mask = (mask > 127).astype(np.uint8)
        
        # Compute distance transform
        # distance_transform_edt gives distance to nearest 0-valued pixel (edge)
        dist_transform = ndimage.distance_transform_edt(mask).astype(np.float32)
        
        # For each fly position, lookup distance to edge
        # Positions are in (x, y) format, but array is indexed as [row, col]
        x_pos = self.dt["x"].values.astype(int)
        y_pos = self.dt["y"].values.astype(int)
        
        # Clip to valid range (in case positions are slightly out of bounds)
        x_pos = np.clip(x_pos, 0, dist_transform.shape[1] - 1)
        y_pos = np.clip(y_pos, 0, dist_transform.shape[0] - 1)
        
        # Lookup distance at each position
        edge_distances = dist_transform[y_pos, x_pos]
        
        self.dt["edge_distance"] = edge_distances
        
        logger.info(f"Computed edge distances: mean={edge_distances.mean():.2f}, "
                    f"std={edge_distances.std():.2f}, "
                    f"min={edge_distances.min():.2f}, "
                    f"max={edge_distances.max():.2f}")
        
    @property
    def number_of_food_blobs(self):
        if self.landmarks is None:
            self.load_landmarks(errors="raise")
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
