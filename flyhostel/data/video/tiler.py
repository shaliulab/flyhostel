import numpy as np
import cv2


class VideoTiler:
    """
    Make a video resulting from displaying several existing videos of equal length and resolution
    """

    def __init__(self, paths, grid):
        self._paths = paths
        assert self._paths, "Please provide at least one video"
        self._grid = grid
        self._video_writer = None
        assert np.prod(self._grid) >= len(self._paths), f"{np.prod(self._grid)} , {len(self._paths)}"

        self._caps = {}

        for path in self._paths:
            self._caps[path] =cv2.VideoCapture(path)

        path = self._paths[0]
        self._frame_size=(
            int(self._caps[path].get(3)) * grid[1],
            int(self._caps[path].get(4)) * grid[0]
        )

        self._is_color = len(self._caps[path].read()[1].shape)==3
        self._caps[path].set(1,0)



    def __call__(self, output, fps=None):

        if fps is None:
            fps = int(self._caps[self._paths[0]].get(5))



        self._video_writer = cv2.VideoWriter(
            output,
            cv2.VideoWriter_fourcc(*"MP4V"),
            fps=fps,
            frameSize=self._frame_size,
            isColor=self._is_color
        )


        ret=True
        while ret:
            arr=[]
            row=[]
            for _ in range(self._grid[1]):
                row.append(None)

            for _ in range(self._grid[0]):
                arr.append(row.copy())


            row_id = 0
            col_id = 0
            for path in self._paths:
                ret, frame = self._caps[path].read()
                if not ret:
                    break
                arr[row_id][col_id]=frame

                col_id+=1

                if col_id == self._grid[1]:
                    col_id=0
                    row_id+=1

            if not ret:
                break

            rows = [np.hstack(row) for row in arr]
            img = np.vstack(rows)
            self._video_writer.write(img)

        self._video_writer.release()
