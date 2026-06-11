from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import pandas as pd
import torch

from graph_builder import build_graph


# ---------------------------------------------------------
# Label setup
# ---------------------------------------------------------
LABEL_COLS = ["scale", "arpeggio", "chord", "jump"]


# ---------------------------------------------------------
# Paths matching your current project structure
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_CSV_PATH = BASE_DIR / "dataset" / "labels" / "musette-in-d-major.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "dataset" / "graphs" / "musette-in-d-major"
    

@dataclass
class ManualNoteRow:
    pitch: int
    onset_beat: float
    duration_beat: float
    voice: int
    measure: int
    y: list[int]
    staff: int = 0

    def __getitem__(self, key: str):
        return getattr(self, key)


def load_manual_notes(csv_path: str | Path) -> list[ManualNoteRow]:
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Manual label CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = [
        "pitch_midi",
        "onset_beat",
        "duration_beat",
        "voice",
        "measure",
        "staff",
        *LABEL_COLS,
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    notes: list[ManualNoteRow] = []

    for _, row in df.iterrows():
        y = [int(row[col]) for col in LABEL_COLS]

        notes.append(
            ManualNoteRow(
                pitch=int(row["pitch_midi"]),
                onset_beat=float(row["onset_beat"]),
                duration_beat=float(row["duration_beat"]),
                voice=int(row["voice"]),
                measure=int(row["measure"]),
                staff=int(row["staff"]),
                y=y,
            )
        )

    return sorted(
        notes,
        key=lambda note: (
            note.onset_beat,
            note.staff,
            note.voice,
            note.pitch,
        ),
    )


def get_context_measures(
    target_measure: int,
    all_measures: list[int],
    context: int,
) -> list[int]:
    measure_set = set(all_measures)
    context_measures = []

    for measure in range(target_measure - context, target_measure + context + 1):
        if measure in measure_set:
            context_measures.append(measure)

    return context_measures


def make_window_graph(
    piece_notes: list[ManualNoteRow],
    piece_id: str,
    target_measure: int,
    all_measures: list[int],
    context: int,
):
    context_measures = get_context_measures(
        target_measure=target_measure,
        all_measures=all_measures,
        context=context,
    )

    context_set = set(context_measures)

    window_notes = [
        note for note in piece_notes
        if note.measure in context_set
    ]

    if not window_notes:
        raise ValueError(
            f"No notes found for piece={piece_id}, target_measure={target_measure}"
        )

    data = build_graph(window_notes)

    data.y = torch.tensor(
        [note.y for note in window_notes],
        dtype=torch.float32,
    )

    data.target_mask = torch.tensor(
        [note.measure == target_measure for note in window_notes],
        dtype=torch.bool,
    )

    data.piece_id = piece_id
    data.window_id = target_measure
    data.target_measure = target_measure
    data.context_measures = context_measures
    data.source = "manual"

    return data


def make_manual_graphs(
    csv_path: str | Path,
    piece_id: str,
    context: int = 1,
) -> list:
    piece_notes = load_manual_notes(csv_path)

    all_measures = sorted({note.measure for note in piece_notes})

    graphs = []

    for target_measure in all_measures:
        graph = make_window_graph(
            piece_notes=piece_notes,
            piece_id=piece_id,
            target_measure=target_measure,
            all_measures=all_measures,
            context=context,
        )

        graphs.append(graph)

    return graphs


def save_manual_graphs(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    piece_id: str = "fur_elise",
    context: int = 1,
    clear_output_dir: bool = True,
) -> list:
    output_path = Path(output_dir)

    if clear_output_dir and output_path.exists():
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    graphs = make_manual_graphs(
        csv_path=csv_path,
        piece_id=piece_id,
        context=context,
    )

    for idx, graph in enumerate(graphs):
        save_path = output_path / f"{piece_id}_manual_{idx:04d}.pt"
        torch.save(graph, save_path)

    print(f"Saved {len(graphs)} manual graphs to {output_path}")

    return graphs


if __name__ == "__main__":
    save_manual_graphs(
        csv_path=DEFAULT_CSV_PATH,
        output_dir=DEFAULT_OUTPUT_DIR,
        piece_id="fur_elise",
        context=1,
    )