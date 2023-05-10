import os.path
import argparse
import cv2
import numpy as np

import sleap
# sleap.disable_preallocation()  # This initializes the GPU and prevents TensorFlow from filling the entire GPU memory
print(sleap.versions())
print(sleap.system_summary())

import pandas as pd
import vidio
import joblib

SLEAP_MODELS_DIR="/staging/leuven/stg_00115/Data/flyhostel_data/flyhostel_deepethogram/models"
CENTROID_MODEL=os.path.join(SLEAP_MODELS_DIR, "230509_095538.centroid.n=308")
CENTERED_INSTANCE_MODEL=os.path.join(SLEAP_MODELS_DIR, "230510_131148.centered_instance.n=485")
print("Centroid model:")
print(f"    {CENTROID_MODEL}")
print("Centered instance model:")
print(f"    {CENTERED_INSTANCE_MODEL}")

NODES=None
HEIGHT=WIDTH=200


def predict_video(video, local_identity, stride, nodes=None, output_file=None):
    predictor = sleap.load_model([CENTROID_MODEL, CENTERED_INSTANCE_MODEL], batch_size=16)
    
    video=vidio.VideoReader(video)
    rois = {local_identity: ((local_identity-1) * WIDTH, 0, WIDTH, HEIGHT)}
    video.load_roi(rois=rois, roi=local_identity)

    # Load frames to a numpy array.
    imgs = [cv2.cvtColor(video[i], cv2.COLOR_BGR2GRAY) for i in range(0, len(video), stride)]
    imgs=np.stack(imgs)
    imgs=imgs.reshape((*imgs.shape, 1))
    print(f"imgs.shape: {imgs.shape}")
    
    # Predict on numpy array.
    predictions = predictions = predictor.predict(imgs)
    skeleton=predictions.skeleton

    output = []
    if nodes is None:
        nodes = [node.name for node in skeleton.nodes]
    for prediction in predictions.predicted_instances:
        for i, node in enumerate(prediction.nodes):
            if node.name in nodes:
                output.append(
                (
                    prediction.frame_idx,
                    node.name,
                    round(prediction.points[i].x),
                    round(prediction.points[i].y),
                    round(prediction.points[i].score, 2),
                )
        )

    if output_file is not None:
        data = pd.DataFrame.from_records(output)
        data.columns=["frame_number", "node", "x", "y", "score"]
        data.to_csv(output_file)

    return output


def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", nargs="+", type=str, required=True, help="List of videos to run inference on")
    ap.add_argument("--local-identity", required=True, type=int, help="Local identity in the video")
    ap.add_argument("--stride", type=int, required=True, help="Downsampling factor of video, so every stride th frame will be processed")
    ap.add_argument("--n-jobs", type=int, default=1, help="How many jobs to run in parallel, default 1")
    return ap


def main():

    ap = get_parser()
    args=ap.parse_args()
    videos=args.video

    joblib.Parallel(args.n_jobs)(
        joblib.delayed(
            predict_video
        )(
            video, args.local_identity,
            stride=args.stride,
            nodes=NODES,
            output_file=os.path.splitext(video)[0] + f"_{str(args.local_identity).zfill(3)}_predictions.csv"
        )
        for video in videos
    )


if __name__ == "__main__":
    main()