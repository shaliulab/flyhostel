import logging
from abc import ABC, abstractmethod
import sqlite3
import os.path
from tqdm.auto import tqdm
from .reader import MP4Reader
import cv2
from flyhostel.utils import get_square_height, get_square_width
logger=logging.getLogger(__name__)

DEBUG=False
FLYHOSTEL_SINGLE_VIDEOS="./flyhostel/single_animal/"

class MP4VideoMaker(ABC):

    _basedir = None
    _number_of_animals = None
    video_writer=None
    experiment=None
    _flyhostel_dataset = None
    _index_db=None
    framerate=None
    start_next_chunk=False
    _identifiers=None
    _stacked=None

    @abstractmethod
    def init_video_writer(self, basedir, frame_size, first_chunk=0, chunksize=None):
        return


    @staticmethod
    def fetch_frame_time(cur, frame_number):
        return
    
    def save_coords_to_csv(self, chunks, output):
        store_path=os.path.join(self._basedir, "metadata.yaml")

        square_width=get_square_width(self.experiment)
        square_height=get_square_height(self.experiment)


        with sqlite3.connect(f"file:{self._flyhostel_dataset}?mode=ro", uri=True) as conn:
            for chunk in chunks:
                with MP4Reader(
                        "flyhostel", connection=conn, store_path=store_path,
                        number_of_animals=self._number_of_animals,
                        width=square_width, height=square_height, resolution=(square_width, square_height),
                        background_color=255, chunks=[chunk]
                    ) as mp4_reader:
                    data=mp4_reader._data.reset_index()

                    for i, identifier in enumerate(self._identifiers):
                        output_folder=os.path.join(FLYHOSTEL_SINGLE_VIDEOS, str(identifier).zfill(3))
                        csv = os.path.join(output_folder, f"{str(chunk).zfill(6)}.csv")
                        df=data.loc[data[mp4_reader.IDENTIFIER_COLUMN]==identifier].set_index([
                            "frame_number",
                            mp4_reader.IDENTIFIER_COLUMN
                        ])
                        os.makedirs(output_folder, exist_ok=True)
                        df.to_csv(csv)

    def _make_single_video(self, chunks, output, frame_size, resolution, background_color=255, **kwargs):
        width, height = frame_size
        store_path=os.path.join(self._basedir, "metadata.yaml")
        capfn=None

        self.video_writer={id: None for id in [0] + list(range(1, self._number_of_animals+1))}
        self.txt_file={id: None for id in [0] + list(range(1, self._number_of_animals+1))}


        with sqlite3.connect(f"file:{self._flyhostel_dataset}?mode=ro", uri=True) as conn:
            with sqlite3.connect(f"file:{self._index_db}?mode=ro", uri=True) as index_conn:
                index_cur = index_conn.cursor()

                for chunk in chunks:
                    target_fn = None

                    count_NULL=0
                    
                    with MP4Reader(
                            "flyhostel", connection=conn, store_path=store_path,
                            number_of_animals=self._number_of_animals,
                            width=width, height=height, resolution=resolution,
                            background_color=background_color, chunks=[chunk]
                        ) as mp4_reader:

                        if not mp4_reader.success:
                            # warnings.warn(f"Skipping chunk {chunk}")
                            raise Exception(f"Cannot fetch data for chunk {chunk}")
                            
                            
                        written_images={identifier: 0 for identifier in [0] + self._identifiers}

                        while True:

                            data = mp4_reader.read(target_fn, self._identifiers, stacked=self._stacked)
                            if data is None:
                                break

                            frame_number, img = data
                            if img is None:
                                break

                            for i, identifier in enumerate(self._identifiers):
                                fn, written_images_=self.write_frame(img[i], output, chunk, frame_number, identifier, resolution, index_cur=index_cur, written_images=written_images[identifier], **kwargs)
                                written_images[identifier]=written_images_
                                if fn is not None:
                                    logger.debug(f"Working on chunk {str(identifier).zfill(3)}/{chunk}. Initialized {fn}. start_next_chunk = {self.start_next_chunk}, chunks={chunks}")
                            
                            target_fn=frame_number+mp4_reader.step


                        if self._stacked:
                            self.video_writer[0].close()

                            with open(self.txt_file[0], "w", encoding="utf8") as filehandle:
                                filehandle.write(f"{written_images[0]}\n")
                            with open("status.txt", "a", encoding="utf8") as filehandle:
                                filehandle.write(f"Chunk {chunk}:{count_NULL}:{written_images[0]}\n")

                        else:
                            with open("status.txt", "a", encoding="utf8") as filehandle:
                                for identifier in self._identifiers:
                                    self.video_writer[identifier].close()
                                    with open(self.txt_file[identifier], "w", encoding="utf8") as filehandle2:
                                        filehandle2.write(f"{written_images[identifier]}\n")
                                    
                                    filehandle.write(f"Chunk {chunk}:{count_NULL}:{written_images[0]}\n")

        return capfn



    def write_frame(self, img, output, chunk, frame_number, identifier, resolution, index_cur, written_images, **kwargs):
        
        paded_identifier=str(identifier).zfill(3)
        
        if self.video_writer[identifier] is None:
            
            output=os.path.join(FLYHOSTEL_SINGLE_VIDEOS, paded_identifier)
            os.makedirs(output, exist_ok=True)
            fn, written_images = self.init_video_writer(basedir=output, frame_size=resolution, identifier=identifier,chunk=chunk, **kwargs)
            if fn is None:
                return fn, written_images
            logger.debug("Working on chunk %s. Initialized %s. start_next_chunk = %s", chunk, fn, self.start_next_chunk)
            assert img.shape == resolution[::-1], f"{img.shape} != {resolution[::-1]}"
            assert str(chunk).zfill(6) in fn

        if DEBUG:
            text = str(frame_number)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2

            # Frame is 100x100
            frame_h, frame_w = 100, 100

            # Get text size
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

            # Compute bottom-left corner so that text is centered
            x = (frame_w - text_w) // 2
            y = (frame_h + text_h) // 2

            cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


        frame_time = self.fetch_frame_time(index_cur, frame_number)
        assert img.shape == resolution[::-1], f"{img.shape} != {resolution[::-1]}"
        capfn=self.video_writer[identifier]._capfn
        fn = self.video_writer[identifier].add_image(
            img, frame_number, frame_time, annotate=False,
            start_next_chunk=self.start_next_chunk
        )

        if DEBUG:
            folder=f"images_{paded_identifier}"
            os.makedirs(folder, exist_ok=True)
            cv2.imwrite(f"{folder}/img_{str(frame_number).zfill(10)}.png", img)

        # pb.update(1)
        if written_images % (self.framerate * 1) == 0:
            with open(self.txt_file[identifier], "w", encoding="utf8") as filehandle:
                filehandle.write(f"{written_images}\n")

        written_images+=1
        return fn, written_images
