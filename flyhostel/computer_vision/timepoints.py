import numpy as np


class TimePoints:

    def __init__(self, timepoints):
        self._timepoints=timepoints
        if type(self._timepoints) is np.ndarray:
            self._type="points"
        elif type(self._timepoints) is tuple:
            self._type="interval"
        else:
            raise Exception(f"timepoints is of type {type(timepoints)}. Please pass a tuple or a np.ndarray")

    def sample(self, n):
        if self._type == "points":
            return self.__class__(self._timepoints[np.linspace(0, len(self._timepoints)-1, n, dtype=np.int64)])
        elif self._type == "interval":
            return self.__class__(np.linspace(*self._timepoints, n, dtype=np.int64))

    def set(self, index):

        if self._type == "interval":
            self._timepoints = index[np.bitwise_and(
                index > self._timepoints[0],
                index < self._timepoints[1]
            )]
            self._type = "points"
        else:
            pass

    def __gt__(self, other):
        return self._timepoints > other

    def __lt__(self, other):
        return self._timepoints < other

    def __ge__(self, other):
        return self._timepoints >= other

    def __le__(self, other):
        return self._timepoints <= other

    def __eq__(self, other):
        return self._timepoints == other

    def __neq__(self, other):
        return self._timepoints != other

    def __add__(self, other):
        return self._timepoints + other

    def __sub__(self, other):
        return self._timepoints - other

    def __getitem__(self, idx):
        # make sure the output is still an array, even if of length 1
        # this is needed to not have an exception thrown in the init
        # because otherwise the type is np.float64 or np.int64 instead of np.array
        # return self.__class__(np.array(self._timepoints[idx]))
        data=self._timepoints[idx]
        if type(data) is np.ndarray:
            return self.__class__(data)

        else:
            return data
        

    def __repr__(self):
        return self._timepoints.__repr__()
    
    def __iter__(self):
        assert self._type == "points"

        for msec in self._timepoints:
            yield msec

    def __len__(self):
        return len(self._timepoints)
