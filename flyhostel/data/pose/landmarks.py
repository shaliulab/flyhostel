"""
Load the coordinates of the landmarks saved in the LANDMARKS table
The landmarks can be 

    * food: an ellipse that delineates the food patch
    * notch: a tetragon that delinates the notches in the glass above the arena
"""
import logging
import sqlite3
import numpy as np
import pandas as pd
from flyhostel.utils import get_dbfile
logger=logging.getLogger(__name__)


import numpy as np
from matplotlib.path import Path

def point_to_line_segment_distance(point, line_start, line_end):
    """
    Computes the distance from a point to a line segment.
    :param point: numpy array of the point [x, y]
    :param line_start: numpy array of the line segment start point [x, y]
    :param line_end: numpy array of the line segment end point [x, y]
    :return: distance from the point to the line segment
    """
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len = np.dot(line_vec, line_vec)
    
    if line_len == 0:
        # line_start and line_end are the same point
        return np.linalg.norm(point - line_start, axis=1)
    
    t = np.clip(np.dot(point_vec, line_vec) / line_len, 0, 1)
    closest_point = line_start + np.outer(t, line_vec)
    
    return np.linalg.norm(point - closest_point, axis=1)

def points_inside_polygon(points, polygon):
    """
    Checks if points are inside a polygon.
    :param points: numpy array of shape (N, 2) where N is the number of points
    :param polygon: list or numpy array of the polygon vertices [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    :return: boolean array indicating if each point is inside the polygon
    """
    path = Path(polygon)
    return path.contains_points(points)

def distance_from_points_to_polygon(points, polygon):
    """
    Computes the distance from an array of points to a polygon with 4 corners.
    :param points: numpy array of shape (N, 2) where N is the number of points
    :param polygon: list or numpy array of the polygon vertices [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    :return: distances from each point to the polygon (negative if inside)
    """
    points = np.array(points)
    polygon = np.array(polygon)
    num_vertices = len(polygon)
    
    min_distances = np.full(points.shape[0], float('inf'))
    
    for i in range(num_vertices):
        line_start = polygon[i]
        line_end = polygon[(i + 1) % num_vertices]
        distances = point_to_line_segment_distance(points, line_start, line_end)
        min_distances = np.minimum(min_distances, distances)
    
    inside = points_inside_polygon(points, polygon)
    min_distances[inside] = -min_distances[inside]
    
    return min_distances


def distance_from_points_to_ellipse(points, h, k, a, b, theta):
    """
    Computes the distance from an array of points to an ellipse.
    :param points: numpy array of shape (N, 2) where N is the number of points
    :param h: x-coordinate of the center of the ellipse
    :param k: y-coordinate of the center of the ellipse
    :param a: semi-major axis of the ellipse
    :param b: semi-minor axis of the ellipse
    :param theta: rotation angle of the ellipse in radians
    :return: distances from each point to the ellipse (negative if inside)
    """
    points = np.array(points)
    
    # Translate points to the center of the ellipse
    x_shifted = points[:, 0] - h
    y_shifted = points[:, 1] - k
    
    # Rotate the points to align with the ellipse axes
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x_rot = cos_theta * x_shifted + sin_theta * y_shifted
    y_rot = -sin_theta * x_shifted + cos_theta * y_shifted
    
    # Calculate normalized distances
    normalized_distances = (x_rot**2 / a**2) + (y_rot**2 / b**2)
    
    # Determine if points are inside the ellipse
    inside = normalized_distances <= 1
    
    # Calculate angles for parametric closest points
    t = np.arctan2(b * y_rot, a * x_rot)
    
    # Parametric coordinates of closest points on the ellipse
    ex = a * np.cos(t)
    ey = b * np.sin(t)
    
    # Calculate Euclidean distances to the ellipse
    distances_to_contour = np.sqrt((x_rot - ex)**2 + (y_rot - ey)**2)
    
    # For points inside the ellipse, distance is negative
    distances = np.where(inside, -distances_to_contour, distances_to_contour)
    
    return distances


def is_point_in_ellipse(x, y, h, k, a, b, theta):
    # Translate point to the center of the ellipse
    x_shifted = x - h
    y_shifted = y - k
    
    # Rotation matrix
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, sin_theta],
                                [-sin_theta, cos_theta]])
    
    # Rotate the point coordinates
    point = np.array([x_shifted, y_shifted])
    x_rot, y_rot = np.dot(rotation_matrix, point)
    
    # Check if the point is within the ellipse
    ellipse_eq = (x_rot**2 / a**2) + (y_rot**2 / b**2)
    return ellipse_eq <= 1



class LandmarksLoader:
    basedir=None
    dt=None

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

        distances=self.dt[[f"food_{i+1}_dist" for i in range(food_blobs.shape[0])]].values
        self.dt["food"]=distances.argmin(axis=1)
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
        self.dt["notch"]=distances.argmin(axis=1)
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

