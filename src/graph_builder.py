# graph_builder.py

import torch
from torch_geometric.data import Data


def build_node_features(note_array):
    """
    Convert notes → node feature matrix
    """

    x = []

    for note in note_array:
        pitch = note["pitch"]
        onset = note["onset_beat"]
        duration = note["duration_beat"]
        voice = note["voice"]

        pitch_class = pitch % 12
        octave = pitch // 12

        # simple placeholder for staff (you can improve later)
        staff_id = 0 if pitch >= 60 else 1

        x.append([
            pitch,
            duration,
            onset,
            staff_id,
            voice,
            pitch_class,
            octave
        ])

    return torch.tensor(x, dtype=torch.float)


def build_edges(note_array):
    """
    Build music-aware edges:
    0 = temporal edge
    1 = simultaneous/chord edge
    2 = same-voice edge
    """

    edge_index = []
    edge_attr = []

    num_notes = len(note_array)
    eps = 1e-5

    def add_edge(i, j, relation_type):
        pitch_i = note_array[i]["pitch"]
        pitch_j = note_array[j]["pitch"]

        onset_i = note_array[i]["onset_beat"]
        onset_j = note_array[j]["onset_beat"]

        voice_i = note_array[i]["voice"]
        voice_j = note_array[j]["voice"]

        staff_i = 0 if pitch_i >= 60 else 1
        staff_j = 0 if pitch_j >= 60 else 1

        delta_onset = onset_j - onset_i
        interval = pitch_j - pitch_i

        same_staff = 1 if staff_i == staff_j else 0
        same_voice = 1 if voice_i == voice_j else 0

        edge_index.append([i, j])
        edge_attr.append([
            relation_type,
            delta_onset,
            interval,
            same_staff,
            same_voice
        ])

    # 1. Temporal edges: note i -> note i+1
    for i in range(num_notes - 1):
        j = i + 1

        add_edge(i, j, relation_type=0)
        add_edge(j, i, relation_type=0)

    # 2. Simultaneous edges: notes starting at same time
    for i in range(num_notes):
        for j in range(i + 1, num_notes):
            onset_i = note_array[i]["onset_beat"]
            onset_j = note_array[j]["onset_beat"]

            if abs(onset_i - onset_j) < eps:
                add_edge(i, j, relation_type=1)
                add_edge(j, i, relation_type=1)

    # 3. Same-voice edges: consecutive notes inside each voice
    voices = {}

    for i, note in enumerate(note_array):
        voice = note["voice"]
        voices.setdefault(voice, []).append(i)

    for voice, note_indices in voices.items():
        # Sort notes in this voice by onset
        note_indices = sorted(
            note_indices,
            key=lambda idx: note_array[idx]["onset_beat"]
        )

        for k in range(len(note_indices) - 1):
            i = note_indices[k]
            j = note_indices[k + 1]

            add_edge(i, j, relation_type=2)
            add_edge(j, i, relation_type=2)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return edge_index, edge_attr


def build_graph(note_array):
    """
    Full graph builder
    """

    x = build_node_features(note_array)
    edge_index, edge_attr = build_edges(note_array)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )

    return data