import pandas as pd
class Behavr(pd.DataFrame):

    def __init__(self, data, metadata, *args, **kwargs):
        super(Behavr, self).__init__(data, *args, **kwargs)
        self._metadata = metadata

        
    def setmeta(self, metadata):
        assert metadata.index.name == "id"
        assert all([id in metadata.index for id in self.index])
        self._metadata = metadata