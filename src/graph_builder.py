import torch
from torch_geometric.data import Data


BEATS_PER_MEASURE = 4.0


def build_node_features(note_array):
    """
    Convert notes → normalized node feature matrix.

    Each note = one node.

    Node features:
    [
        pitch / 127,
        duration / 4,
        beat_in_measure / 4,
        voice / 10,
        pitch_class / 11,
        octave / 10
    ]
    """

    x = []

    for note in note_array:
        pitch = float(note["pitch"])
        onset = float(note["onset_beat"])
        duration = float(note["duration_beat"])
        voice = float(note["voice"])

        pitch_class = pitch % 12
        octave = pitch // 12

        # IMPORTANT:
        # Do NOT use absolute onset like 200, 300, 400.
        # Use position inside the measure instead.
        beat_in_measure = onset % BEATS_PER_MEASURE

        x.append(
            [
                pitch / 127.0,
                duration / BEATS_PER_MEASURE,
                beat_in_measure / BEATS_PER_MEASURE,
                voice / 10.0,
                pitch_class / 11.0,
                octave / 10.0,
            ]
        )

    return torch.tensor(x, dtype=torch.float)


def build_edges(note_array):
    """
    Build music-aware edges:

    0 = temporal edge
    1 = simultaneous/harmonic edge
    2 = same-voice edge

    Normalized edge attributes:
    [
        relation_type / 3,
        delta_onset / 4,
        interval / 24,
        same_voice
    ]
    """

    edge_index = []
    edge_attr = []

    num_notes = len(note_array)
    eps = 1e-5

    def add_edge(i, j, relation_type):
        pitch_i = float(note_array[i]["pitch"])
        pitch_j = float(note_array[j]["pitch"])

        onset_i = float(note_array[i]["onset_beat"])
        onset_j = float(note_array[j]["onset_beat"])

        voice_i = int(note_array[i]["voice"])
        voice_j = int(note_array[j]["voice"])

        delta_onset = onset_j - onset_i
        interval = pitch_j - pitch_i

        same_voice = 1.0 if voice_i == voice_j else 0.0

        edge_index.append([i, j])
        edge_attr.append(
            [
                relation_type / 3.0,
                delta_onset / BEATS_PER_MEASURE,
                interval / 24.0,
                same_voice,
            ]
        )

    # 1. Temporal edges: note i <-> note i+1
    for i in range(num_notes - 1):
        j = i + 1
        add_edge(i, j, relation_type=0)
        add_edge(j, i, relation_type=0)

    # 2. Simultaneous edges: notes starting at same time
    for i in range(num_notes):
        for j in range(i + 1, num_notes):
            onset_i = float(note_array[i]["onset_beat"])
            onset_j = float(note_array[j]["onset_beat"])

            if abs(onset_i - onset_j) < eps:
                add_edge(i, j, relation_type=1)
                add_edge(j, i, relation_type=1)

    # 3. Same-voice edges: consecutive notes inside each voice
    voices = {}

    for i, note in enumerate(note_array):
        voice = int(note["voice"])
        voices.setdefault(voice, []).append(i)

    for voice, note_indices in voices.items():
        note_indices = sorted(
            note_indices,
            key=lambda idx: float(note_array[idx]["onset_beat"]),
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
    Full graph builder.
    """

    x = build_node_features(note_array)
    edge_index, edge_attr = build_edges(note_array)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )

    return data