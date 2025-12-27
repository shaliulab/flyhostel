import logging
import traceback
import os.path
import json

import pandas as pd
import cv2
import hashlib

from imgstore.interface import VideoCapture
from idtrackerai_validator_server.backend import draw_frame, annotate_frame

logger=logging.getLogger(__name__)

def get_checksum(path):
    return hashlib.md5(open(path,'rb').read()).hexdigest()



class ImageWriter:
    header=[
        {"version":"1.0"},
        {"type":"images"}
    ]

    def __init__(self, basedir, fn0, extension):
        self._basedir=basedir
        self._count=0
        self._fn0=fn0
        self._extension=extension
        os.makedirs(self._basedir, exist_ok=True)
        self.image_manifest=[]

    def write(self, frame):
        
        dest_path=os.path.join(
            self._basedir,
            os.path.basename(self._basedir.rstrip(os.path.sep)) + "_" + str(self._fn0 + self._count) + self._extension
        )
        cv2.imwrite(
            dest_path,
            frame
        )
        self.add_to_manifest(dest_path)
        self._count+=1

    def release(self):
        manifest_file=os.path.join(
            os.path.dirname(self._basedir),
            os.path.basename(self._basedir) + ".jsonl"
        )

        self.write_manifest(manifest_file)

    def add_to_manifest(self, path):
        self.image_manifest.append(
            self.generate_image_manifest(path)
        )

    def generate_image_manifest(self, path):
        image=cv2.imread(path)
        height, width=image.shape[:2]
        checksum=get_checksum(path)
        name, extension=os.path.splitext(os.path.basename(path))

        return {
            "name": os.path.join(os.path.basename(self._basedir), name),
            "extension": extension,
            "width": width,
            "height": height,
            "meta":{"related_images":[]},
            "checksum":checksum
        }



    def write_manifest(self, path):
        manifest=self.header + self.image_manifest

        with open(path, 'w') as file:
            for item in manifest:
                json_str = json.dumps(item)
                file.write(json_str + '\n')


def generate_validation_video(store_path, row, df, number_of_animals, chunksize, framerate, output_folder=".", format=".mp4", field="identity"):
    vw=None
    cap=None
    output_video=None
    output_path_csv=None

    try:
        cap=VideoCapture(store_path, 50)
        frame_number_0=row["frame_number"]
        frame_number_last=df["frame_number"].iloc[-1]

        experiment=row["experiment"]
        try:
            cap.set(1, frame_number_0)
        except Exception as error:
            logger.error("Cannot set FRAME_POS to %s on video %s", frame_number_0, store_path)
            raise error


        chunk=frame_number_0//chunksize
        output_path_csv=os.path.join(output_folder, f"{experiment}_{str(chunk).zfill(6)}_{frame_number_0}.csv")
        accum=0

        while True:
            ret, frame = cap.read()
            
            if not ret or cap.frame_number>frame_number_last:
                break

            frame_number=cap.frame_number
            tracking_data=df.loc[df["frame_number"]==frame_number]
            frame=draw_frame(frame, tracking_data, number_of_animals, field=field)
            frame=annotate_frame(frame, row)

            if vw is None:
                suffix=str(frame_number_0//chunksize).zfill(6) + "_" + str(frame_number_0).zfill(10)
                if format==".mp4":

                    output_video=os.path.join(output_folder, f"{experiment}_{suffix}.mp4")
                    logger.debug("Saving %s", output_video)
                    vw=cv2.VideoWriter(
                        output_video,
                        cv2.VideoWriter_fourcc(*"MP4V"),
                        frameSize=frame.shape[:2][::-1],
                        fps=framerate,
                        isColor=True
                    )
                elif format in [".png", ".jpg"]:
                    output_video=os.path.join(output_folder,  f"{experiment}_{suffix}")
                    logger.debug("Saving %s", output_video)

                    vw=ImageWriter(
                        output_video,
                        fn0=frame_number_0,
                        extension=format
                    )

            vw.write(frame)
            accum+=1

        logger.info("Wrote %s frames", accum)

        if output_video is not None and os.path.exists(output_video):
            pd.DataFrame(row).T.to_csv(output_path_csv)

    except Exception as error:
        logger.error(error)
        logger.error(traceback.print_exc())

    finally:
        if vw is not None:
            vw.release()

        if cap is not None:
            cap.release()

    return output_video
