import math
import plotly.express as px


def draw_umap(dataset, max_points=50_000):
    
    if dataset.shape[0] > max_points:
        skip = math.ceil(dataset.shape[0] / max_points)
        dataset=dataset.iloc[::skip]

    fig=px.scatter(
        dataset, x="C_1", y="C_2", color="behavior",
        hover_data=["id", "chunk", "frame_idx", "zt", "behavior"],
    )

    return fig