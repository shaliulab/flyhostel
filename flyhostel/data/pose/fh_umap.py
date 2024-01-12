import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

id_cols=["id", "frame_number", "t", "frame_idx", "chunk"]
label_cols=["behavior"]


def add_n_steps_in_the_past(ml_datasets, feature_cols, n_steps=1):
    ml_datasets_steps=[]
    trainable_columns_steps=[]
    for id, df in ml_datasets.groupby("id"):
        df_id=None
        for step in range(n_steps):
            pad_row=[np.nan for _ in range(len(feature_cols))]
            values=df[feature_cols].iloc[step:].values
            if step >0 :
                pad_rows=np.vstack([pad_row for _ in range(step)])
                print(pad_rows.shape)
                values=np.vstack([pad_rows, values])
            columns=[feat + "_" + str(step) for feat in feature_cols]
            trainable_columns_steps.extend(columns)
            df_step=pd.DataFrame(values, columns=columns)
            if df_id is None:
                df_id = df_step.iloc[n_steps:]
            else:
                df_id=pd.concat([df_id, df_step.iloc[n_steps:]], axis=1)

        for col in id_cols + label_cols:
            if col in df.columns:
                df_id[col]=np.nan
                df_id[col]=df[col].iloc[n_steps:].values
 
        ml_datasets_steps.append(df_id)
    ml_datasets_steps=pd.concat(ml_datasets_steps, axis=0)
    trainable_columns_steps=list(set(trainable_columns_steps))
    return ml_datasets_steps, trainable_columns_steps