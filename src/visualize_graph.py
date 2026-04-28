# visualize_graph.py

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx


def visualize_music_graph(data, save_path: str | None = None) -> None:
    """
    Save or show a PyTorch Geometric graph in a music-like layout.

    Assumes:
    - data.x[:, 0] = pitch_midi
    - data.x[:, 2] = onset_in_measure
    """

    graph_nx = to_networkx(data, to_undirected=False)

    pos = {}
    labels = {}

    num_nodes = data.x.size(0)
    for node_id in range(num_nodes):
        pitch = float(data.x[node_id][0].item())
        onset = float(data.x[node_id][2].item())

        # x-axis = onset, y-axis = pitch
        pos[node_id] = (onset, pitch)
        labels[node_id] = f"{node_id}\n{int(pitch)}"

    plt.figure(figsize=(12, 6))

    # --- Split edges by type ---
    edge_index = data.edge_index.numpy()
    edge_attr = data.edge_attr.numpy()

    temporal_edges = []
    simultaneous_edges = []
    voice_edges = []

    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        relation = int(edge_attr[i][0])

        if relation == 0:
            temporal_edges.append((src, dst))
        elif relation == 1:
            simultaneous_edges.append((src, dst))
        elif relation == 2:
            voice_edges.append((src, dst))


    # --- Draw nodes ---
    nx.draw_networkx_nodes(graph_nx, pos, node_size=500)

    # --- Draw edges with colors ---
    nx.draw_networkx_edges(graph_nx, pos, edgelist=temporal_edges, edge_color='black')
    nx.draw_networkx_edges(graph_nx, pos, edgelist=simultaneous_edges, edge_color='red')
    nx.draw_networkx_edges(graph_nx, pos, edgelist=voice_edges, edge_color='blue')

    # --- Draw labels ---
    nx.draw_networkx_labels(graph_nx, pos, labels=labels, font_size=6)

    plt.xlabel("Onset in measure")
    plt.ylabel("Pitch (MIDI)")
    plt.title("Music Graph")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()