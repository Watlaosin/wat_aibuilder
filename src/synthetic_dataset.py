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

# Simple fake hand / voice convention.
# This is NOT staff. It is just a voice/register convention for synthetic data.
RIGHT_VOICE = 1
LEFT_VOICE = 5

RIGHT_BASE_PITCHES = [60, 62, 64, 65, 67, 69]
LEFT_BASE_PITCHES = [36, 40, 43, 48, 50, 52]

HAND_TO_VOICE = {
    "right": RIGHT_VOICE,
    "left": LEFT_VOICE,
}


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
        onset_beat=round(onset_beat, 3),
        duration_beat=round(duration_beat, 3),
        voice=voice,
        measure=measure,
        y=LABEL_TO_Y[label],
    )


def _choose_hand(rng: random.Random) -> str:
    return rng.choice(["right", "left"])


def _base_pitch_for_hand(rng: random.Random, hand: str) -> int:
    if hand == "right":
        return rng.choice(RIGHT_BASE_PITCHES)
    return rng.choice(LEFT_BASE_PITCHES)


def _add_opposite_hand_support(
    notes: list[FakeNoteRow],
    rng: random.Random,
    measure: int,
    technique_hand: str,
) -> None:
    """
    Add accompaniment in the opposite hand, labeled none.

    This teaches:
    LH/RH support + technique notes does NOT automatically mean block chord.
    """
    start = measure * BEATS_PER_MEASURE

    support_hand = "left" if technique_hand == "right" else "right"
    support_voice = HAND_TO_VOICE[support_hand]

    if support_hand == "left":
        bass = rng.choice([36, 40, 43, 48, 52])
        support_pitches = [bass, bass + 7]
    else:
        top = rng.choice([60, 62, 64, 67])
        support_pitches = [top]

    for pitch in support_pitches:
        notes.append(
            _make_note(
                pitch=pitch,
                onset_beat=start,
                duration_beat=4.0,
                voice=support_voice,
                measure=measure,
                label="none",
            )
        )


def _generate_scale_measure(rng: random.Random, measure: int) -> list[FakeNoteRow]:
    start = measure * BEATS_PER_MEASURE

    hand = _choose_hand(rng)
    voice = HAND_TO_VOICE[hand]
    base_pitch = _base_pitch_for_hand(rng, hand)

    scale_templates = [
        [0, 2, 4, 5, 7, 9, 11, 12],  # major
        [0, 2, 3, 5, 7, 8, 10, 12],  # natural minor
        [0, 1, 2, 3, 4, 5, 6, 7],    # chromatic fragment
    ]

    intervals = rng.choice(scale_templates)

    if rng.random() < 0.5:
        intervals = list(reversed(intervals))

    notes: list[FakeNoteRow] = []

    if rng.random() < 0.8:
        _add_opposite_hand_support(notes, rng, measure, hand)

    for i, interval in enumerate(intervals):
        notes.append(
            _make_note(
                pitch=base_pitch + interval,
                onset_beat=start + i * 0.5,
                duration_beat=0.5,
                voice=voice,
                measure=measure,
                label="scale",
            )
        )

    return notes


def _generate_arpeggio_measure(rng: random.Random, measure: int) -> list[FakeNoteRow]:
    start = measure * BEATS_PER_MEASURE

    hand = _choose_hand(rng)
    voice = HAND_TO_VOICE[hand]
    root = _base_pitch_for_hand(rng, hand)

    arpeggio_templates = [
        [0, 4, 7, 12, 7, 4, 0, 4],    # major
        [0, 3, 7, 12, 7, 3, 0, 3],    # minor
        [0, 4, 7, 10, 12, 10, 7, 4],  # dominant 7 style
    ]

    intervals = rng.choice(arpeggio_templates)

    if rng.random() < 0.5:
        intervals = list(reversed(intervals))

    notes: list[FakeNoteRow] = []

    if rng.random() < 0.8:
        _add_opposite_hand_support(notes, rng, measure, hand)

    for i, interval in enumerate(intervals):
        notes.append(
            _make_note(
                pitch=root + interval,
                onset_beat=start + i * 0.5,
                duration_beat=0.5,
                voice=voice,
                measure=measure,
                label="arpeggio",
            )
        )

    return notes


def _generate_chord_measure(rng: random.Random, measure: int) -> list[FakeNoteRow]:
    """
    Generate block-chord technique.

    This means notes intentionally pressed together as a block,
    not just harmonic overlap between LH and RH.
    """
    start = measure * BEATS_PER_MEASURE

    hand_mode = rng.choice(["right", "left", "both"])
    chord_quality = rng.choice(["major", "minor", "diminished"])

    if chord_quality == "major":
        intervals = [0, 4, 7]
    elif chord_quality == "minor":
        intervals = [0, 3, 7]
    else:
        intervals = [0, 3, 6]

    beats = rng.choice([
        [0.0, 2.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0],
    ])

    if len(beats) == 1:
        duration = 4.0
    elif len(beats) == 2:
        duration = 2.0
    else:
        duration = 1.0

    if hand_mode == "both":
        hands = ["left", "right"]
    else:
        hands = [hand_mode]

    notes: list[FakeNoteRow] = []

    for hand in hands:
        voice = HAND_TO_VOICE[hand]
        root = _base_pitch_for_hand(rng, hand)

        for beat in beats:
            for interval in intervals:
                notes.append(
                    _make_note(
                        pitch=root + interval,
                        onset_beat=start + beat,
                        duration_beat=duration,
                        voice=voice,
                        measure=measure,
                        label="chord",
                    )
                )

    return notes


def _generate_jump_measure(rng: random.Random, measure: int) -> list[FakeNoteRow]:
    start = measure * BEATS_PER_MEASURE

    hand = _choose_hand(rng)
    voice = HAND_TO_VOICE[hand]
    base_pitch = _base_pitch_for_hand(rng, hand)

    is_jump_arpeggio = rng.random() < 0.25
    label = "jump_arpeggio" if is_jump_arpeggio else "jump"

    if is_jump_arpeggio:
        intervals = [0, 12, 4, 16, 7, 19, 12, 24]
    else:
        intervals = rng.choice([
            [0, 12, 0, 12, -2, 10, -4, 8],
            [0, 13, -2, 15, 1, -12, 5, 18],
            [0, 9, -1, 10, -2, 11, -3, 12],
        ])

    notes: list[FakeNoteRow] = []

    if rng.random() < 0.8:
        _add_opposite_hand_support(notes, rng, measure, hand)

    for i, interval in enumerate(intervals):
        notes.append(
            _make_note(
                pitch=base_pitch + interval,
                onset_beat=start + i * 0.5,
                duration_beat=0.5,
                voice=voice,
                measure=measure,
                label=label,
            )
        )

    return notes


def _generate_none_measure(rng: random.Random, measure: int) -> list[FakeNoteRow]:
    """
    Generate neutral notes with no target technique.

    This includes simple textures similar to easy piano arrangements:
    held bass + simple melody, repeated notes, sparse notes, etc.
    """
    start = measure * BEATS_PER_MEASURE
    notes: list[FakeNoteRow] = []

    texture = rng.choice([
        "held_bass",
        "repeated_note",
        "simple_two_hand",
        "sparse_melody",
    ])

    if texture == "held_bass":
        bass = rng.choice([36, 40, 43, 48, 52])
        for pitch in [bass, bass + 7]:
            notes.append(
                _make_note(
                    pitch=pitch,
                    onset_beat=start,
                    duration_beat=4.0,
                    voice=LEFT_VOICE,
                    measure=measure,
                    label="none",
                )
            )

    elif texture == "repeated_note":
        hand = _choose_hand(rng)
        voice = HAND_TO_VOICE[hand]
        pitch = _base_pitch_for_hand(rng, hand)

        for beat in [0.0, 1.0, 2.0, 3.0]:
            notes.append(
                _make_note(
                    pitch=pitch,
                    onset_beat=start + beat,
                    duration_beat=1.0,
                    voice=voice,
                    measure=measure,
                    label="none",
                )
            )

    elif texture == "simple_two_hand":
        # Similar to easy piano texture:
        # LH support + RH simple melody, but no special technique label.
        bass = rng.choice([36, 40, 43, 48, 52])
        melody = rng.choice([60, 62, 64, 65, 67])

        notes.append(
            _make_note(
                pitch=bass,
                onset_beat=start,
                duration_beat=4.0,
                voice=LEFT_VOICE,
                measure=measure,
                label="none",
            )
        )

        for beat in [0.0, 1.0, 2.0, 3.0]:
            notes.append(
                _make_note(
                    pitch=melody + rng.choice([-1, 0, 1]),
                    onset_beat=start + beat,
                    duration_beat=1.0,
                    voice=RIGHT_VOICE,
                    measure=measure,
                    label="none",
                )
            )

    else:
        hand = _choose_hand(rng)
        voice = HAND_TO_VOICE[hand]
        pitch = _base_pitch_for_hand(rng, hand)

        selected_beats = rng.sample([0.0, 1.0, 2.0, 3.0], rng.randint(1, 3))

        for beat in selected_beats:
            notes.append(
                _make_note(
                    pitch=pitch + rng.choice([-2, 0, 2]),
                    onset_beat=start + beat,
                    duration_beat=1.0,
                    voice=voice,
                    measure=measure,
                    label="none",
                )
            )

    return notes


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