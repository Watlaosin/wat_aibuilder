# src/parser.py

import partitura as pt


def load_score(file_path):
    score = pt.load_score(file_path)
    return score


def extract_notes(score):
    note_array = score.note_array()

    print("Available note fields:", note_array.dtype.names)

    # Prefer beat-based duration if available
    if "duration_beat" in note_array.dtype.names:
        note_array = note_array[note_array["duration_beat"] > 0]
    elif "duration_div" in note_array.dtype.names:
        note_array = note_array[note_array["duration_div"] > 0]

    return note_array