import torch
from torch_geometric.data import Data


BEATS_PER_MEASURE = 4.0


def get_note_value(note, key, default=0):
    if isinstance(note, dict):
        return note.get(key, default)

    if hasattr(note, "dtype") and note.dtype.names and key in note.dtype.names:
        value = note[key]
        return value.item() if hasattr(value, "item") else value

    try:
        return note[key]
    except Exception:
        return default


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
        staff / 2,
        pitch_class / 11,
        octave / 10
    ]
    """

    x = []

    for note in note_array:
        pitch = float(get_note_value(note, "pitch", 60))
        onset = float(get_note_value(note, "onset_beat", 0.0))
        duration = float(get_note_value(note, "duration_beat", 1.0))
        voice = float(get_note_value(note, "voice", 0))
        staff = float(get_note_value(note, "staff", 0))

        pitch_class = pitch % 12
        octave = pitch // 12

        # Do NOT use absolute onset like 200, 300, 400.
        # Use position inside the measure instead.
        beat_in_measure = onset % BEATS_PER_MEASURE

        x.append(
            [
                pitch / 127.0,
                duration / BEATS_PER_MEASURE,
                beat_in_measure / BEATS_PER_MEASURE,
                voice / 10.0,
                staff / 2.0,
                pitch_class / 11.0,
                octave / 10.0,
            ]
        )

    return torch.tensor(x, dtype=torch.float)


def build_edges(note_array):
    """
    Build music-aware edges:

    0 = temporal edge within same staff
    1 = simultaneous/harmonic edge
    2 = same-voice edge

    Normalized edge attributes:
    [
        relation_type / 3,
        delta_onset / 4,
        interval / 24,
        same_voice,
        same_staff
    ]
    """

    edge_index = []
    edge_attr = []

    num_notes = len(note_array)
    eps = 1e-5

    def add_edge(i, j, relation_type):
        pitch_i = float(get_note_value(note_array[i], "pitch", 60))
        pitch_j = float(get_note_value(note_array[j], "pitch", 60))

        onset_i = float(get_note_value(note_array[i], "onset_beat", 0.0))
        onset_j = float(get_note_value(note_array[j], "onset_beat", 0.0))

        voice_i = int(get_note_value(note_array[i], "voice", 0))
        voice_j = int(get_note_value(note_array[j], "voice", 0))

        staff_i = int(get_note_value(note_array[i], "staff", 0))
        staff_j = int(get_note_value(note_array[j], "staff", 0))

        delta_onset = onset_j - onset_i
        interval = pitch_j - pitch_i

        same_voice = 1.0 if voice_i == voice_j else 0.0
        same_staff = 1.0 if staff_i != 0 and staff_i == staff_j else 0.0

        edge_index.append([i, j])
        edge_attr.append(
            [
                relation_type / 3.0,
                delta_onset / BEATS_PER_MEASURE,
                interval / 24.0,
                same_voice,
                same_staff,
            ]
        )

    # 1. Temporal edges: consecutive notes inside each staff.
    # This avoids connecting LH bass notes directly to RH melody notes.
    staff_groups = {}

    for i, note in enumerate(note_array):
        staff = int(get_note_value(note, "staff", 0))
        staff_groups.setdefault(staff, []).append(i)

    for staff, note_indices in staff_groups.items():
        note_indices = sorted(
            note_indices,
            key=lambda idx: (
                float(get_note_value(note_array[idx], "onset_beat", 0.0)),
                float(get_note_value(note_array[idx], "pitch", 60)),
            ),
        )

        for k in range(len(note_indices) - 1):
            i = note_indices[k]
            j = note_indices[k + 1]

            add_edge(i, j, relation_type=0)
            add_edge(j, i, relation_type=0)

    # 2. Simultaneous edges: notes starting at the same time.
    # These can cross staves because harmony/chords often involve both hands.
    for i in range(num_notes):
        for j in range(i + 1, num_notes):
            onset_i = float(get_note_value(note_array[i], "onset_beat", 0.0))
            onset_j = float(get_note_value(note_array[j], "onset_beat", 0.0))

            if abs(onset_i - onset_j) < eps:
                add_edge(i, j, relation_type=1)
                add_edge(j, i, relation_type=1)

    # 3. Same-voice edges: consecutive notes inside each voice.
    # This is stricter than same-staff temporal edges.
    voices = {}

    for i, note in enumerate(note_array):
        voice = int(get_note_value(note, "voice", 0))
        voices.setdefault(voice, []).append(i)

    for voice, note_indices in voices.items():
        note_indices = sorted(
            note_indices,
            key=lambda idx: float(get_note_value(note_array[idx], "onset_beat", 0.0)),
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