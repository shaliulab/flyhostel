import logging
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
logger=logging.getLogger(__name__)

def apply_validation_csv_file(new_data, machine_data, validation_csv, chunksize):
    extra_rows=[]
    #columns
    # frame_number  in_frame_index  local_identity  validated  fragment           x           y  modified class_name  chunk

    manual_validation=pd.read_csv(validation_csv)
    for _, manual_validation in tqdm(manual_validation.iterrows(), desc="Applying manual validation", total=manual_validation.shape[0]):
        frame_number=manual_validation["frame_number"]
        chunk=frame_number//chunksize
        fragment=manual_validation["fragment"]
        replace=manual_validation["replace"]
        local_identity=manual_validation["local_identity"]

        if replace:
            if manual_validation.get("by_identity", True):
                extra_data=new_data.loc[((new_data["frame_number"]==frame_number)&(new_data["local_identity"]==local_identity))]
                extra_data["fragment"]=np.nan
            else:
                if np.isnan(manual_validation.get("first_frame_number", np.nan)):
                    extra_data=new_data.loc[((new_data["frame_number"]==frame_number)&(new_data["fragment"]==fragment))]
                else:
                    frame_numbers=np.arange(manual_validation["first_frame_number"], manual_validation["last_frame_number"]+1)
                    if np.isnan(fragment):
                        extra_data=pd.DataFrame({"frame_number": frame_numbers})
                        extra_data["fragment"]=np.nan
                        extra_data["chunk"]=chunk
                        extra_data["x"]=manual_validation["x"]
                        extra_data["y"]=manual_validation["y"]
                        extra_data["local_identity"]=local_identity
                        extra_data["is_a_crossing"]=False
                        extra_data["validated"]=1
                        extra_data["in_frame_index"]=np.nan
                        extra_data["modified"]=1
                        extra_data["class_name"]=None                                    
                        extra_data["frame_validated"]=False
                        index=((new_data["frame_number"].isin(frame_numbers))&(new_data["local_identity"]==local_identity))
                        nrows_removed=index.sum()
                        print(f"Removing {nrows_removed} rows")
                        new_data=new_data.loc[~index]
                        extra_rows.append(extra_data)
                        continue

                    else:
                        extra_data=new_data.loc[((new_data["frame_number"].isin(frame_numbers))&(new_data["fragment"]==fragment))]

            extra_data["in_frame_index"]=np.nan
            nrows=new_data.shape[0]
 
            foo=new_data.merge(extra_data[[]], left_index=True, right_index=True, how="outer", indicator=True)
            new_data=foo.loc[foo["_merge"]=="left_only"].drop("_merge", axis=1)
            new_nrows=new_data.shape[0]
            assert nrows-new_nrows==extra_data.shape[0]
            logger.info("Modified %s rows of dataset", extra_data.shape[0])
            del foo

            extra_data["local_identity"]=local_identity
            extra_data["is_a_crossing"]=False
            extra_data["validated"]=1
            extra_data["frame_validated"]=False
            extra_rows.append(extra_data)


        else:
            extra_data=machine_data.loc[(machine_data["chunk"]==chunk)]
            if np.isnan(manual_validation.get("first_frame_number", np.nan)):
                frame_numbers=[frame_number]
            else:
                frame_numbers=np.arange(manual_validation["first_frame_number"], manual_validation["last_frame_number"]+1)
            
            for frame_number in frame_numbers:
                if np.isnan(fragment):
                    ref_data=extra_data.loc[extra_data["frame_number"]==frame_number]
                    extra_data=pd.DataFrame({
                        "frame_number": [frame_number],
                        "in_frame_index": [ref_data["in_frame_index"].max()+1],
                        "fragment": [ref_data["fragment"].max()+1],
                        "modified": [1],
                        "class_name": [None],
                        "chunk": [chunk]
                    })
                else:
                    extra_data=extra_data.loc[(extra_data["fragment"]==fragment)]

                extra_data["x"]=manual_validation["x"]
                extra_data["y"]=manual_validation["y"]

                extra_data["local_identity"]=local_identity
                extra_data["is_a_crossing"]=False
                extra_data["validated"]=1
                extra_data["frame_validated"]=False
                extra_rows.append(extra_data)

    new_data.reset_index(drop=True, inplace=True)
    if extra_rows:
        extra_data=pd.concat(extra_rows, axis=0).reset_index(drop=True)
        new_data=pd.concat([
            new_data,
            extra_data[new_data.columns]
        ], axis=0).reset_index(drop=True).sort_values(["frame_number", "local_identity"])
    
    return new_data
