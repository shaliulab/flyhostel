import sqlite3
from tqdm.auto import tqdm

from flyhostel.data.video.reader import MP4Reader
STORE_PATH="/lustre1/project/stg_00115/Data/flyhostel_data/videos/FlyHostel3/6X/2023-02-05_17-00-00/metadata.yaml"

WIDTH=200
HEIGHT=200
RESOLUTION=(200, 200)

def test_yolov7_mode():
    reader=MP4Reader.from_store_path(
        "yolov7",
        store_path=STORE_PATH,
        chunks=[100],
        width=WIDTH, height=HEIGHT, resolution=RESOLUTION,
    )

    for path, img, im0s, vid_cap in tqdm(reader, unit=f"batch (size: {MP4Reader.BATCH_SIZE})"):

        assert len(path) == len(im0s)
        # print(img.shape)
        for i in range(len(path)):
            pass
            # print(path[i])
            # print(im0s[i].shape)

        print(reader.frame)

def test_flyhostel_mode():

    flyhostel_dataset="/lustre1/project/stg_00115/Data/flyhostel_data/videos/FlyHostel3/6X/2023-02-05_17-00-00/FlyHostel3_6X_2023-02-05_17-00-00.db"
    number_of_animals=6
    frame_number=None

    pb=tqdm(unit="frame")


    with sqlite3.connect(flyhostel_dataset, check_same_thread=False) as conn:

        with MP4Reader(
            "flyhostel", connection=conn, store_path=STORE_PATH, number_of_animals=number_of_animals,
            width=WIDTH, height=HEIGHT, resolution=RESOLUTION,
            chunks=[100]
        ) as mp4_reader:

            while True:
                data = mp4_reader.read(frame_number, number_of_animals, stack=True)
                if data is None:
                    break
                pb.update(1)

                frame_number, img = data

        pb.close()

# test_flyhostel_mode()
test_yolov7_mode()