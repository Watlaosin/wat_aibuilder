from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import shutil

import torch

from graph_builder import build_graph


LABEL_NAMES = ["scale", "arpeggio", "chord", "jump"]
LABEL_TO_Y = {
    "scale": [1, 0, 0, 0],
    "arpeggio": [0, 1, 0, 0],
    "chord": [0, 0, 1, 0],
    "jump": [0, 0, 0, 1],
    "jump_arpeggio": [0, 1, 0, 1],
    "none": [0, 0, 0, 0],
}

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "dataset" / "synthetic_graphs"
BEATS_PER_MEASURE = 4.0


@dataclass
class FakeNoteRow:
    pitch: int
    onset_beat: float
    duration_beat: float
    voice: int
    measure: int
    y: list[int]

    def __getitem__(self, key: str):
        return getattr(self, key)


def _make_note(
    pitch: int,
    onset_beat: float,
    duration_beat: float,
    voice: int,
    measure: int,
    label: str,
) -> FakeNoteRow:
    return FakeNoteRow(
        pitch=pitch,
        onset_beat=onset_beat,
        duration_beat=duration_beat,
        voice=voice,
        measure=measure,
        y=LABEL_TO_Y[label],
    )


def _generate_scale_measure(rng: random.Random, measure: int) -> list[FakeNoteRow]:
    start = measure * BEATS_PER_MEASURE
    base_pitch = rng.choice([55, 57, 60, 62, 64])
    direction = rng.choice([-1, 1])

    return [
        _make_note(
            pitch=base_pitch + direction * i,
            onset_beat=start + i * 0.5,
            duration_beat=0.5,
            voice=0,
            measure=measure,
            label="scale",
        )
        for i in range(8)
    ]


def _generate_arpeggio_measure(rng: random.Random, measure: int) -> list[FakeNoteRow]:
    start = measure * BEATS_PER_MEASURE
    root = rng.choice([48, 50, 52, 53, 55, 57, 60])
    chord_tones = [0, 4, 7, 12, 7, 4, 0, 4]

    return [
        _make_note(
            pitch=root + chord_tones[i],
            onset_beat=start + i * 0.5,
            duration_beat=0.5,
            voice=0,
            measure=measure,
            label="arpeggio",
        )
        for i in range(8)
    ]


def _generate_chord_measure(rng: random.Random, measure: int) -> list[FakeNoteRow]:
    start = measure * BEATS_PER_MEASURE
    root = rng.choice([48, 50, 52, 53, 55, 57, 60])
    chord = [root, root + 4, root + 7]
    notes: list[FakeNoteRow] = []

    for beat in (0.0, 2.0):
        for voice, pitch in enumerate(chord):
            notes.append(
                _make_note(
                    pitch=pitch,
                    onset_beat=start + beat,
                    duration_beat=2.0,
                    voice=voice,
                    measure=measure,
                    label="chord",
                )
            )

    return notes


def _generate_jump_measure(rng: random.Random, measure: int) -> list[FakeNoteRow]:
    start = measure * BEATS_PER_MEASURE
    base_pitch = rng.choice([48, 50, 52, 55, 57, 60])
    is_jump_arpeggio = rng.random() < 0.3
    label = "jump_arpeggio" if is_jump_arpeggio else "jump"

    if is_jump_arpeggio:
        intervals = [0, 12, 4, 16, 7, 19, 12, 24]
    else:
        intervals = [0, 13, -2, 15, 1, -12, 5, 18]

    return [
        _make_note(
            pitch=base_pitch + intervals[i],
            onset_beat=start + i * 0.5,
            duration_beat=0.5,
            voice=0,
            measure=measure,
            label=label,
        )
        for i in range(8)
    ]


def _generate_none_measure(rng: random.Random, measure: int) -> list[FakeNoteRow]:
    start = measure * BEATS_PER_MEASURE
    pitch = rng.choice([55, 57, 60, 62, 64])

    return [
        _make_note(
            pitch=pitch + rng.choice([0, 0, 0, 1, -1]),
            onset_beat=start + beat,
            duration_beat=1.0,
            voice=0,
            measure=measure,
            label="none",
        )
        for beat in (0.0, 1.0, 2.0, 3.0)
    ]


def generate_synthetic_piece(num_measures: int, rng: random.Random) -> list[FakeNoteRow]:
    measure_generators = {
        "scale": _generate_scale_measure,
        "arpeggio": _generate_arpeggio_measure,
        "chord": _generate_chord_measure,
        "jump": _generate_jump_measure,
        "none": _generate_none_measure,
    }

    notes: list[FakeNoteRow] = []

    for measure in range(num_measures):
        pattern = rng.choice(list(measure_generators))
        notes.extend(measure_generators[pattern](rng, measure))

    return sorted(notes, key=lambda note: (note.onset_beat, note.voice, note.pitch))


def _context_measures(target_measure: int, num_measures: int, context: int) -> list[int]:
    start = max(0, target_measure - context)
    end = min(num_measures - 1, target_measure + context)
    return list(range(start, end + 1))


def _make_window_graph(
    piece_notes: list[FakeNoteRow],
    piece_id: int,
    target_measure: int,
    num_measures: int,
    context: int,
):
    context_measures = _context_measures(target_measure, num_measures, context)
    context_set = set(context_measures)
    window_notes = [note for note in piece_notes if note.measure in context_set]

    data = build_graph(window_notes)

    data.y = torch.tensor([note.y for note in window_notes], dtype=torch.float32)
    data.target_mask = torch.tensor(
        [note.measure == target_measure for note in window_notes],
        dtype=torch.bool,
    )

    data.piece_id = piece_id
    data.window_id = target_measure
    data.target_measure = target_measure
    data.context_measures = context_measures

    return data


def make_synthetic_dataset(
    num_pieces: int = 100,
    measures_per_piece: int = 8,
    context: int = 1,
    seed: int = 42,
) -> list:
    rng = random.Random(seed)
    dataset = []

    for piece_id in range(num_pieces):
        piece_notes = generate_synthetic_piece(measures_per_piece, rng)

        for target_measure in range(measures_per_piece):
            dataset.append(
                _make_window_graph(
                    piece_notes=piece_notes,
                    piece_id=piece_id,
                    target_measure=target_measure,
                    num_measures=measures_per_piece,
                    context=context,
                )
            )

    return dataset


def save_synthetic_dataset(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> list:
    output_path = Path(output_dir)

    if output_path.exists():
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    dataset = make_synthetic_dataset()

    for idx, data in enumerate(dataset):
        torch.save(data, output_path / f"synth_{idx:05d}.pt")

    return dataset


if __name__ == "__main__":
    saved_dataset = save_synthetic_dataset()
    print(f"Saved {len(saved_dataset)} graphs to {DEFAULT_OUTPUT_DIR}")