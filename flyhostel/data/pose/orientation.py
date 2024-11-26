import numpy as np

def compute_angles(data):
    """
    Compute angle between horizontal and points provided as input
    Origin is at 50 50
    """

    # Define the fixed points
    origin = np.array([50, 50])
    east = np.array([100, 50])
    
    # Vectors from the origin to the observations
    vectors = data - origin
    
    # Vector from the origin to the east point
    east_vector = east - origin
    
    #    the dot product between the east vector and each observation vector
    dot_products = np.dot(vectors, east_vector)
    
    # Compute the magnitudes of the vectors
    east_magnitude = np.linalg.norm(east_vector)
    vector_magnitudes = np.linalg.norm(vectors, axis=1)
    
    # Compute the cosines of the angles
    cos_angles = dot_products / (east_magnitude * vector_magnitudes)
    
    # Compute the angles in radians
    angles_radians = np.arccos(cos_angles)

    # Compute the cross product to determine the sign of the angle
    cross_products = np.cross(east_vector, vectors)
    
    # Adjust the angles based on the sign of the cross product
    angles_radians = np.where(cross_products < 0, -angles_radians, angles_radians)

    # Convert the angles to degrees
    angles_degrees = np.degrees(angles_radians)
    return angles_degrees


def compute_orientation(pose):
    data=pose[["head_x", "head_y"]].values
    pose["orientation"]=compute_angles(data)
    return pose