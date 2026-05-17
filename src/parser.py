# src/parser.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import partitura as pt


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "pre-annotated-datasets"


def load_score(file_path: str | Path):
    score = pt.load_score(str(file_path))
    return score


def extract_notes(score):
    """
    Extract notes from Partitura.

    We try include_staff=True first.
    If the installed Partitura version does not support it,
    we fall back to normal note_array().
    """
    try:
        note_array = score.note_array(include_staff=True)
    except TypeError:
        note_array = score.note_array()

    print("Available note fields:", note_array.dtype.names)

    # Prefer beat-based duration if available.
    if "duration_beat" in note_array.dtype.names:
        note_array = note_array[note_array["duration_beat"] > 0]
    elif "duration_div" in note_array.dtype.names:
        note_array = note_array[note_array["duration_div"] > 0]

    return note_array


def _field_value(note: Any, field: str, default: Any) -> Any:
    if hasattr(note, "dtype") and note.dtype.names and field in note.dtype.names:
        value = note[field].item() if hasattr(note[field], "item") else note[field]
        return value

    return default


def notes_to_json_rows(note_array) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for index, note in enumerate(note_array):
        onset = float(
            _field_value(
                note,
                "onset_beat",
                _field_value(note, "onset_div", index),
            )
        )

        duration = float(
            _field_value(
                note,
                "duration_beat",
                _field_value(note, "duration_div", 1),
            )
        )

        pitch = int(_field_value(note, "pitch", 60))
        voice = int(_field_value(note, "voice", 0))
        staff = int(_field_value(note, "staff", 0))

        if staff == 1:
            hand = "right"
        elif staff == 2:
            hand = "left"
        else:
            hand = "unknown"

        rows.append(
            {
                "note_id": index,
                "pitch": pitch,
                "onset_beat": onset,
                "duration_beat": duration,
                "voice": voice,
                "staff": staff,
                "hand": hand,
                "label": [0, 0, 0, 0],
            }
        )

    return rows


def save_parsed_json(file_path: Path, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    score = load_score(file_path)
    notes = extract_notes(score)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{file_path.stem}.json"

    sample = {
        "id": file_path.stem,
        "source_file": str(file_path),
        "notes": notes_to_json_rows(notes),
    }

    output_path.write_text(
        json.dumps(sample, indent=2) + "\n",
        encoding="utf-8",
    )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse a score into frontend-loadable JSON."
    )
    parser.add_argument(
        "score",
        type=Path,
        help="Path to a MusicXML, compressed MXL, or supported score file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    output_path = save_parsed_json(args.score, args.output_dir)
    print(f"Saved parsed JSON: {output_path}")


if __name__ == "__main__":
    main()