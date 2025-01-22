import io
import os
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.express.colors import sample_colorscale
from typing import Optional, Any, Tuple
from cycler import cycler


def draw_events(
    hits: np.ndarray[(Any, 3), np.float32],
    labels: np.ndarray[Any, np.int32],
    vertices: Optional[np.ndarray[(Any, 3), np.float32]] = None,
    fakes: Optional[np.ndarray[(Any, 3), np.float32]] = None,
    x_coord_range: Tuple[float, float] = (-851., 851.),
    y_coord_range: Tuple[float, float] = (-851., 851.),
    z_coord_range: Tuple[float, float] = (-2386., 2386.),
    colorscale: str = "Plotly3"
) -> go.Figure:

    fig = go.Figure()
    uniq_labels = np.unique(labels)
    colors = sample_colorscale(colorscale, uniq_labels / uniq_labels.max())

    for i, label in enumerate(uniq_labels):
        event_hits = hits[labels == label]
        fig.add_trace(go.Scatter3d(
            x=event_hits[:, 0],
            y=event_hits[:, 1],
            z=event_hits[:, 2],
            marker=dict(
                size=1,
                color=colors[i],
            ),
            mode="markers",
            name=f"Event #{label}"
        ))

    if vertices is not None:
        fig.add_trace(go.Scatter3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            marker=dict(
                size=2,
                color=colors,
            ),
            mode="markers",
            name="Vertices"
        ))

    if fakes is not None:
        fig.add_trace(go.Scatter3d(
            x=fakes[:, 0],
            y=fakes[:, 1],
            z=fakes[:, 2],
            marker=dict(
                size=1,
                color='gray',
            ),
            opacity=0.35,
            mode='markers',
            name='Fakes',
        ))

    fig.update_layout(
        margin=dict(
            t=20,
            b=10,
            l=10,
            r=10
        ),
        scene=dict(
            xaxis=dict(range=x_coord_range),
            yaxis=dict(range=y_coord_range),
            zaxis=dict(range=z_coord_range),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),
        ),
        scene_camera=dict(
            eye=dict(x=2, y=0.1, z=0.1)
        )
    )

    return fig


def draw_embeddings(
    embeddings: np.ndarray[(Any, Any), np.float32],
    labels: np.ndarray[Any, np.int32],
    sample_idx: int,
    split_name: str
):
    label_set = np.unique(labels)
    num_classes = len(label_set)
    plt.figure(figsize=(7, 7))
    plt.gca().set_prop_cycle(
        cycler(
            "color", [plt.cm.nipy_spectral(i)
                      for i in np.linspace(0, 0.9, num_classes)]
        )
    )
    for i in range(num_classes):
        idx = labels == label_set[i]
        plt.plot(embeddings[idx, 0], embeddings[idx, 1], ".", markersize=5)
    plt.title(f"UMAP plot for the #{sample_idx} {split_name} sample")
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    return plt


def visualize_embeddings_eval(
        ax,
        embeddings: np.ndarray,
        labels: np.ndarray,
        sample_idx: int,
        split_name: str
):
    label_set = np.unique(labels)
    num_classes = len(label_set)
    ax.set_prop_cycle(
        cycler(
            "color", [plt.cm.nipy_spectral(i)
                      for i in np.linspace(0, 0.9, num_classes)]
        )
    )
    for i in range(num_classes):
        idx = labels == label_set[i]
        ax.plot(embeddings[idx, 0], embeddings[idx, 1], ".", markersize=5)
    ax.set_title(f"UMAP plot for the #{sample_idx} {split_name} sample")
