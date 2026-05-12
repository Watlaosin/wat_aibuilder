# src/parser.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import partitura as pt
import xml.etree.ElementTree as ET


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "pre-annotated-datasets"


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

def extract_staff_by_note_id(file_path: str | Path) -> dict[str, int]:
    """
    Extract MusicXML <staff> for notes using each note's id attribute.

    Returns:
        {
            "note-id": staff_number
        }

    If a note has no <staff>, it is skipped.
    """
    staff_by_id: dict[str, int] = {}

    tree = ET.parse(file_path)
    root = tree.getroot()

    # Handle MusicXML namespaces if present.
    if root.tag.startswith("{"):
        namespace = root.tag.split("}")[0].strip("{")
        ns = {"mx": namespace}
        note_path = ".//mx:note"
        staff_path = "mx:staff"
    else:
        ns = {}
        note_path = ".//note"
        staff_path = "staff"

    for note_el in root.findall(note_path, ns):
        note_id = note_el.attrib.get("id")
        if not note_id:
            continue

        staff_el = note_el.find(staff_path, ns)
        if staff_el is None or staff_el.text is None:
            continue

        try:
            staff_by_id[note_id] = int(staff_el.text)
        except ValueError:
            continue

    return staff_by_id


def _field_value(note: Any, field: str, default: Any) -> Any:
    if field in note.dtype.names:
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

        rows.append(
            {
                "note_id": index,
                "pitch": pitch,
                "onset_beat": onset,
                "duration_beat": duration,
                "voice": voice,
                "hand": "right" if pitch >= 60 else "left",
                "label": [0, 0, 0, 0],
            }
        )

    return rows


def save_parsed_json(file_path: Path, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    score = load_score(str(file_path))
    notes = extract_notes(score)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{file_path.stem}.json"
    sample = {
        "id": file_path.stem,
        "source_file": str(file_path),
        "notes": notes_to_json_rows(notes),
    }
    output_path.write_text(json.dumps(sample, indent=2) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a score into frontend-loadable JSON.")
    parser.add_argument("score", type=Path, help="Path to a MusicXML, compressed MXL, or supported score file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_path = save_parsed_json(args.score, args.output_dir)
    print(f"Saved parsed JSON: {output_path}")


if __name__ == "__main__":
    main()
