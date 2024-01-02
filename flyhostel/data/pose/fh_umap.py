import numpy as np
import pandas as pd

id_cols=["id", "frame_number", "t", "frame_idx", "chunk"]
label_cols=["behavior"]



class PreprocessUMAP:
    
    @staticmethod
    def compute_distance_features(pose, part1, part2):
        distance=np.sqrt(
            (pose[f"{part1}_x"]-pose[f"{part2}_x"])**2 + (pose[f"{part1}_y"]-pose[f"{part2}_y"])**2
        )
        return distance


    def obtain_umap_input(self, pose_dt0, pose_dt1, dt0_features=["head_proboscis_distance"]):
        
        pose=pose_dt1.copy()
        parts=[("head", "proboscis"),]

        distance_features={
            f"{part1}_{part2}_distance": self.compute_distance_features(pose_dt0, part1, part2)
            for part1, part2 in parts
        }

        distance_features=pd.DataFrame(distance_features)
        distance_features["t"]=pose_dt0["t"]
        distance_features["id"]=pose_dt0["id"]
        distance_features["frame_number"]=pose_dt0["frame_number"]
        
        pose=pose.merge(distance_features, on=["id", "frame_number", "t"], how="left")
        return pose

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