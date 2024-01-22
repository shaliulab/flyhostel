import codetiming
import numpy as np
import pandas as pd

from flyhostel.data.interactions.distance import compute_distance_matrix_bodyparts


def compute_pairwise_distances_using_bodyparts_cpu(self, distances, pose, bodyparts, precision=100):
        """
        Compute distance between two closest bodyparts of two already close animals
        See compute_distance_matrix_bodyparts
        """
   
        bodyparts_arr = np.array(bodyparts)
        distance_matrix, bp_pairs = compute_distance_matrix_bodyparts(distances, pose, np, bodyparts, precision=precision)
        with codetiming.Timer(text="Computing closest body parts spent: {:.2f} seconds"):
            selected_bp_pairs = np.argmin(distance_matrix, axis=1)
        min_distance=(distance_matrix[np.arange(distance_matrix.shape[0]), selected_bp_pairs]/precision)

        pairs=np.stack([bodyparts_arr[bp_pairs[selected]] for selected in selected_bp_pairs], axis=0)
        min_distance_mm = min_distance / self.px_per_mm

        results=pd.DataFrame({"bp_distance_mm": min_distance_mm, "bp_distance": min_distance, "bp1": pairs[:, 0], "bp2": pairs[:, 1]})
        distances = pd.concat([distances.reset_index(drop=True), results], axis=1)
        distances["id"]=pd.Categorical(distances["id"])
        distances["nn"]=pd.Categorical(distances["nn"])
        distances.sort_values(["frame_number", "id", "nn"], inplace=True)
        return distances
