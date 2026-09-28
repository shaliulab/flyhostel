import logging
import os.path
import pandas as pd
from flyhostel.data.pose.proboscis_extension.label_overrides import read_pe_bouts

logger=logging.getLogger(__name__)

class PELoader:
    """
    A class to load the result of the PE pipeline
    """
    
    datasetnames=[]
    basedir=None
    experiment=None
    ids=None
    dt=None

    def __init__(self, *args, **kwargs):
        self.pe=None
        super(PELoader, self).__init__(*args, **kwargs)

    
    def load_centroid_data(*args, **kwargs):
        raise NotImplementedError


    def load_pe_data(
            self,
            min_time=None, max_time=None,
            errors="raise",
            **kwargs,
        ):

        """
        Arguments:
        
        Populates self.pe

        Returns
            None
        """
        pe_dir=f"{self.basedir}/flyhostel/proboscis_extensions/pe_bouts"
        if not os.path.exists(pe_dir):
            msg=f"{pe_dir} not found"
            if errors=="raise":
                raise FileNotFoundError(msg)
            else:
                logger.warning(msg)
                return
        
        pe_trace = f"{pe_dir}/{self.datasetnames[0]}_pe_bouts.feather"
        
        
        if not os.path.exists(pe_trace):
            logger.error("%s not found", pe_trace)
            return None
        

        data=read_pe_bouts(pe_trace)
        
        if self.dt is None:
            self.load_centroid_data(cache="/flyhostel_data/cache")

        data=data.query("label == 'pe'")
        data=data.merge(self.dt[["frame_number", "t"]], how="left", on="frame_number")
        
        if min_time is not None:
            data=data.query("t >= @min_time")
            
        if max_time is not None:
            data=data.query("t < @max_time")

        data["zt"]=(data["t"]//3600)
        data.insert(0, "experiment", self.experiment)
        data.insert(0, "id", self.ids[0])

        for col_name in kwargs:
            if col_name not in data.columns:
                logger.warning("%s not in data", col_name)
                continue

            data = data.query(
                f"`{col_name}` >= @kwargs_value",
                local_dict={"kwargs_value": kwargs[col_name]}
            )

        self.pe=data
