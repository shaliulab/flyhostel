from abc import ABC, abstractmethod
import sqlite3
import os.path
from tqdm.auto import tqdm
from .reader import MP4Reader
FLYHOSTEL_SINGLE_VIDEOS="./flyhostel/single_animal/"

class MP4VideoMaker(ABC):

    _basedir = None
    _number_of_animals = None
    video_writer=None
    _flyhostel_dataset = None
    _index_db=None
    framerate=None
    start_next_chunk=False

    @abstractmethod
    def init_video_writer(self, basedir, frame_size, first_chunk=0, chunksize=None):
        return


    @staticmethod
    def fetch_frame_time(cur, frame_number):
        return

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

                    written_images=0
                    count_NULL=0
                    
                    with MP4Reader(
                            "flyhostel", connection=conn, store_path=store_path,
                            number_of_animals=self._number_of_animals,
                            width=width, height=height, resolution=resolution,
                            background_color=background_color, chunks=[chunk]
                        ) as mp4_reader:
                            
                        resolution_full=(resolution[0] * self._number_of_animals, resolution[1])


                        while True:

                            data = mp4_reader.read(target_fn, self._identifiers, stacked=self._stacked)
                            if data is None:
                                break

                            frame_number, img = data
                            if img is None:
                                break

                            if self._stacked:
                                fn, written_images=self.write_frame(img, output, chunk, frame_number, 0, resolution_full, index_cur=index_cur, written_images=written_images, **kwargs)
                                if fn is not None:
                                    print(f"Working on chunk 000/{chunk}. Initialized {fn}. start_next_chunk = {self.start_next_chunk}, chunks={chunks}")


                            else:
                                for i, identifier in enumerate(self._identifiers):
                                    fn, written_images=self.write_frame(img[i], output, chunk, frame_number, identifier, resolution, index_cur=index_cur, written_images=written_images, **kwargs)
                                    if fn is not None:
                                        print(f"Working on chunk {str(identifier).zfill(3)}/{chunk}. Initialized {fn}. start_next_chunk = {self.start_next_chunk}, chunks={chunks}")

                            target_fn=frame_number+mp4_reader.step


                        if self.stacked:
                            self.video_writer[0].close()

                            with open(self.txt_file[0], "w", encoding="utf8") as filehandle:
                                filehandle.write(f"{written_images}\n")

                        else:
                            for identifier in self._identifiers:
                                self.video_writer[identifier].close()
                                with open(self.txt_file[identifier], "w", encoding="utf8") as filehandle:
                                    filehandle.write(f"{written_images}\n")

                    with open("status.txt", "a", encoding="utf8") as filehandle:
                        filehandle.write(f"Chunk {chunk}:{count_NULL}:{written_images}\n")

        return capfn



    def write_frame(self, img, output, chunk, frame_number, identifier, resolution, index_cur, written_images, **kwargs):
        if self.video_writer[identifier] is None:
            output=os.path.join(FLYHOSTEL_SINGLE_VIDEOS, str(identifier).zfill(3))
            os.makedirs(output, exist_ok=True)
            fn, written_images = self.init_video_writer(basedir=output, frame_size=resolution, identifier=identifier,chunk=chunk, **kwargs)
            if fn is None:
                return fn, written_images
            print(f"Working on chunk {chunk}. Initialized {fn}. start_next_chunk = {self.start_next_chunk}")
            assert img.shape == resolution[::-1], f"{img.shape} != {resolution[::-1]}"
            assert str(chunk).zfill(6) in fn

        frame_time = self.fetch_frame_time(index_cur, frame_number)
        assert img.shape == resolution[::-1], f"{img.shape} != {resolution[::-1]}"
        capfn=self.video_writer[identifier]._capfn
        fn = self.video_writer[identifier].add_image(
            img, frame_number, frame_time, annotate=False,
            start_next_chunk=self.start_next_chunk
        )


        # pb.update(1)
        if written_images % (self.framerate * 1) == 0:
            with open(self.txt_file[identifier], "w", encoding="utf8") as filehandle:
                filehandle.write(f"{written_images}\n")

        written_images+=1
        return fn, written_images
