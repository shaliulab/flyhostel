def select_bounding_box(img, centroid, body_length):
    
    bbox = [
        centroid[0]-body_length//2,
        centroid[1]-body_length//2,
        centroid[0]+body_length//2,
        centroid[1]+body_length//2,
    ]
    
    return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]