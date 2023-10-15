import os.path
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd

def draw_pose_on_axis(pose_data, fns, h5inds, params, pad=80):
    chunksize=45000

    skeleton = params["skeleton"]
    labels=params["labels"]
    body_parts_chosen=params["body_parts_chosen"]
    
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    
    colors=["black", "red", "blue"]


    frame_number=fns["raw"]
    chunk = frame_number // chunksize

    animal=params["animals"][h5inds[0]]
    tokens = animal.split("_")
    prefix=os.path.sep.join(tokens[:2] + ["_".join(tokens[2:4])])
   
    video_path = f"{os.environ['FLYHOSTEL_VIDEOS']}/{prefix}/{str(chunk).zfill(6)}.mp4"
    cap = cv2.VideoCapture(video_path)
    
    frame_idx = frame_number % chunksize
    
    cap.set(1, frame_idx+5)
    ret, frame = cap.read()
    cap.release()
    centroids=[]
    
    for h5ind in h5inds:
    
        centroid = pose_data[h5ind].loc[frame_number, pd.IndexSlice[:, "centroid", :]]
        centroids.append(centroid.values)
    
    mean_centroid = np.stack(centroids).mean(axis=0).astype(np.int64)
    x0 = mean_centroid[0] - pad
    y0 = mean_centroid[1] - pad
    x1 = mean_centroid[0] + pad
    y1 = mean_centroid[1] + pad
    
    xlim = [x0, x1]
    ylim = [y0, y1]
    ax.imshow(frame)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.axis('off')
    
    for h5ind in h5inds:
        centroid = pose_data[h5ind].loc[frame_number, pd.IndexSlice[:, "centroid", :]].values
        
        try:
            color = colors[h5ind]
        except IndexError:
            color = "k"
    
        for edge in skeleton:
            edge_ = [params["body_parts_chosen"][e] for e in edge]
            
            x = (pose_data[h5ind].loc[frame_number, pd.IndexSlice[:, edge_, "x"]]) + centroid[0] - 50
            y = (pose_data[h5ind].loc[frame_number, pd.IndexSlice[:, edge_, "y"]]) + centroid[1] - 50
            ax.plot(x, y, color, lw=1)
            # print(x, y)
    
        bodyparts_index=pose_data[h5ind].columns.names.index("bodyparts")
        assert bodyparts_index == 1
        bodyparts=pose_data[h5ind].columns.get_level_values(bodyparts_index)
    
        for bp_index, bp in enumerate(bodyparts):
            if bodyparts[bp_index] not in body_parts_chosen:
                continue
    
            text = labels[body_parts_chosen.index(bodyparts[bp_index])]
            x = np.round(pose_data[h5ind].loc[frame_number, pd.IndexSlice[:, bp, "x"]].values.flatten()).astype(np.int64) + centroid[0] - 50
            y = np.round(pose_data[h5ind].loc[frame_number, pd.IndexSlice[:, bp, "y"]].values.flatten()).astype(np.int64) + centroid[1] - 50
    
            # ax.text(
            #     x,
            #     y,
            #     text,
            #     color='k'
            # 
    return fig
