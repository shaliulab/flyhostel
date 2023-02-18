from abc import ABC, abstractmethod
import sqlite3
import os.path
from tqdm.auto import tqdm
from .reader import MP4Reader


class MP4VideoMaker(ABC):

    _basedir = None
    _number_of_animals = None
    video_writer=None
    _flyhostel_dataset = None
    _index_db=None
    framerate=None

    @abstractmethod
    def init_video_writer(self, basedir, frame_size, first_chunk=0, chunksize=None):
        return


    @staticmethod
    def fetch_frame_time(cur, frame_number):
        return

    def _make_single_video(self, chunks, output, frame_size, resolution, background_color=255, **kwargs):
        width, height = frame_size
        store_path=os.path.join(self._basedir, "metadata.yaml")

        if output is None:
            output = os.path.join(self._basedir, "flyhostel", "single_animal")

        if not os.path.exists(output):
            os.makedirs(output)


        with sqlite3.connect(f"file:{self._flyhostel_dataset}?mode=ro", uri=True) as conn:
            with sqlite3.connect(f"file:{self._index_db}?mode=ro", uri=True) as index_conn:
                index_cur = index_conn.cursor()

                for chunk in chunks:
                    target_fn = None

                    written_images=0
                    count_NULL=0
                    start_next_chunk=False

                    with MP4Reader(
                            "flyhostel", connection=conn, store_path=store_path,
                            number_of_animals=self._number_of_animals,
                            width=width, height=height, resolution=resolution,
                            background_color=background_color, chunks=[chunk]
                        ) as mp4_reader:

                        # pb = tqdm(unit="frame")

                        while True:

                            data = mp4_reader.read(target_fn, self._number_of_animals, stack=True)
                            if data is None:
                                break

                            frame_number, img = data
                            if img is None:
                                break

                            if self.video_writer is None:
                                resolution_full=(resolution[0] * self._number_of_animals, resolution[1])
                                fn = self.init_video_writer(basedir=output, frame_size=resolution_full, **kwargs)
                                print(f"Working on chunk {chunk}. Initialized {fn}. start_next_chunk = {start_next_chunk}")
                                assert img.shape == resolution_full[::-1]
                                assert str(chunk).zfill(6) in fn

                            frame_time = self.fetch_frame_time(index_cur, frame_number)
                            assert img.shape == resolution_full[::-1], f"{img.shape} != {resolution_full[::-1]}"
                            capfn=self.video_writer._capfn
                            fn = self.video_writer.add_image(
                                img, frame_number, frame_time, annotate=False,
                                start_next_chunk=start_next_chunk
                            )


                            # pb.update(1)
                            if written_images % (self.framerate * 1) == 0:
                                txt_file = f"{os.path.splitext(capfn)[0]}.txt"
                                with open(txt_file, "w", encoding="utf8") as filehandle:
                                    filehandle.write(f"{written_images}\n")

                            written_images+=1
                            target_fn=frame_number+1
                            if fn is not None:
                                print(f"Working on chunk {chunk}. Initialized {fn}. start_next_chunk = {start_next_chunk}, chunks={chunks}")

                            # if written_images == 160*10:
                            #     break

                        self.video_writer.close()
                        with open(txt_file, "w", encoding="utf8") as filehandle:
                            filehandle.write(f"{written_images}\n")

                    with open("status.txt", "a", encoding="utf8") as filehandle:
                        filehandle.write(f"Chunk {chunk}:{count_NULL}:{written_images}\n")

        return capfn
