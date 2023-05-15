import os.path
import warnings

from vidio.read import VideoReader
import cv2

def build_key(flyhostel_id, number_of_animals, date_time, chunk, local_identity=None):
    key = f"FlyHostel{flyhostel_id}_{number_of_animals}X_{date_time}_{str(chunk).zfill(6)}"
    if local_identity is not None:
        key += f"_{str(local_identity).zfill(3)}"

    return key


def parse_key(key):
    tokens = key.split("_")
    return {
        "flyhostel_id": int(tokens[0].replace("FlyHostel", "")),
        "number_of_animals": int(tokens[1].replace("X", "")),
        "date_time": "_".join(tokens[2:4]),
        "chunk": int(tokens[4]),
        "local_identity": int(tokens[5]),
    }



def make_video(scene, color, iterations=1):
    key=build_key(scene["flyhostel_id"], scene["number_of_animals"], scene["datetime"], scene["chunk"], scene["local_identity"])
    filename = f"{key}.mp4"

    filename = os.path.join(os.environ["DEEPETHOGRAM_DATA"], key, filename)
    if not os.path.exists(filename):
        warnings.warn(f"{filename} not found (behavior={scene['behavior']})")
        return

    reader=VideoReader(filename=filename)

    start_fn = int(scene["t_start"] * reader.fps)
    end_fn = int(scene["t_end_video"] * reader.fps)


    assert start_fn < end_fn

    label = scene["text"]
    behavior = scene["text"]

    video_writer = cv2.VideoWriter(
        os.path.join("videos", f"{behavior}_{key}.mp4"),
        cv2.VideoWriter_fourcc(*"MP4V"),
        fps=int(reader.fps),
        frameSize=reader.roi[2:],
        isColor=True
    )

    for _ in range(iterations):
        frame_number=start_fn
        while frame_number < end_fn:
            frame = reader[frame_number]

            x = int(frame.shape[1] * 0.1)
            y = int(frame.shape[0] * 0.2)

            frame = cv2.putText(
                frame, label, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                thickness=3,
                lineType=cv2.LINE_AA
            )

            video_writer.write(frame)
            frame_number +=1

    video_writer.release()
