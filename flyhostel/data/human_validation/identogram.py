import plotly.graph_objects as go
from idtrackerai.utils.py_utils import get_spaced_colors_util

def generate_hover_text(df, meta_columns=["identity","frame_idx", "t_round", "chunk", "frame_number", "fragment"]):
    text = []
    metadata=[df[c] for c in meta_columns]
    for meta in zip(*metadata):
        row=""
        for i, col in enumerate(meta):
            row+=f"{meta_columns[i]}: {col} "
        text.append(row)
    return text
        
def draw_identogram(df_bin, number_of_animals):

    colors=get_spaced_colors_util(number_of_animals, black=False)
    colors={i+1: colors[i] for i in range(len(colors))}
    colors={k: "rgba("+ ", ".join([str(round(e)) for e in v[:3]]) + ", 1)" for k, v in colors.items()}

    fig = go.Figure()

    for label in df_bin["label"].unique():
        filtered_df = df_bin[df_bin["label"] == label]
        text=generate_hover_text(filtered_df)
    
        local_identity=filtered_df["local_identity"].iloc[0].item()
        identity=filtered_df["identity"].iloc[0].item()
        if local_identity==0:
            color="rgba(0, 0, 0, 1)"
        else:
            color=colors[identity]
    
        fig.add_trace(go.Scattergl(
            x=filtered_df['frame_number'],
            y=filtered_df["label"],
            mode='markers',
            marker=dict(size=10, color=[color, ] * filtered_df.shape[0], symbol="square"),
            name=label,
            text=text,
            hoverinfo='text',
        ))


    meta_columns=["frame_idx", "t_round", "chunk", "frame_number"]
    filtered_df=df_bin.loc[df_bin["qc"]==False]
    filtered_df=filtered_df[meta_columns].drop_duplicates()
    color="rgba(0, 0, 0, 1)"
    label="qc-fail"

    text=generate_hover_text(filtered_df, meta_columns)
    fig.add_trace(go.Scattergl(
        x=filtered_df['frame_number'],
        y=["qc-fail", ]*filtered_df.shape[0],
        mode='markers',
        marker=dict(size=10, color=[color, ] * filtered_df.shape[0], symbol="square"),
        name=label,
        text=text,
        hoverinfo='text',
    ))

    # Update layout
    fig.update_layout(
        title="Scatter Plot",
        xaxis_title="Frame Number",
        yaxis_title="Identity/Fragment",
        legend_title="Legend"
    )
    return fig