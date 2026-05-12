from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv

from parser import load_score, extract_notes
from graph_builder import build_graph


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "best_synthetic_gine.pt"
OUTPUT_DIR = BASE_DIR / "predictions"

LABEL_NAMES = ["scale", "arpeggio", "chord", "jump"]
BEATS_PER_MEASURE = 4.0


class TechniqueGINE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        edge_dim: int,
        hidden_channels: int = 64,
        out_channels: int = 4,
    ):
        super().__init__()

        self.node_encoder = nn.Linear(in_channels, hidden_channels)

        nn1 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        nn2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        self.conv1 = GINEConv(nn1, edge_dim=edge_dim)
        self.conv2 = GINEConv(nn2, edge_dim=edge_dim)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)

        return self.classifier(x)


def get_note_value(note, field: str, default=None):
    if field in note.dtype.names:
        value = note[field]
        return value.item() if hasattr(value, "item") else value
    return default


def midi_to_name(pitch: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[pitch % 12]}{pitch // 12 - 1}"


def infer_measure_ids(note_array):
    measure_ids = []

    for note in note_array:
        onset = float(get_note_value(note, "onset_beat", 0.0))
        measure_ids.append(int(onset // BEATS_PER_MEASURE))

    return measure_ids


def context_measures(
    target_measure: int,
    min_measure: int,
    max_measure: int,
    context: int = 1,
):
    start = max(min_measure, target_measure - context)
    end = min(max_measure, target_measure + context)
    return list(range(start, end + 1))


def predict_windowed(score_path: Path, threshold: float = 0.5, context: int = 1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    score = load_score(str(score_path))
    note_array = extract_notes(score)

    if len(note_array) == 0:
        raise ValueError("No notes found in score.")

    measure_ids = infer_measure_ids(note_array)
    min_measure = min(measure_ids)
    max_measure = max(measure_ids)

    full_data = build_graph(note_array)

    model = TechniqueGINE(
        in_channels=full_data.x.shape[1],
        edge_dim=full_data.edge_attr.shape[1],
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    final_probs = torch.zeros((len(note_array), len(LABEL_NAMES)), dtype=torch.float32)

    with torch.no_grad():
        for target_measure in range(min_measure, max_measure + 1):
            ctx = set(
                context_measures(
                    target_measure=target_measure,
                    min_measure=min_measure,
                    max_measure=max_measure,
                    context=context,
                )
            )

            window_indices = [
                i for i, measure in enumerate(measure_ids)
                if measure in ctx
            ]

            if not window_indices:
                continue

            window_note_array = note_array[window_indices]
            window_measure_ids = [measure_ids[i] for i in window_indices]

            data = build_graph(window_note_array).to(device)

            logits = model(data.x, data.edge_index, data.edge_attr)
            probs = torch.sigmoid(logits).cpu()

            for local_idx, original_idx in enumerate(window_indices):
                if window_measure_ids[local_idx] == target_measure:
                    final_probs[original_idx] = probs[local_idx]

    rows = []

    for i, note in enumerate(note_array):
        pitch = int(get_note_value(note, "pitch", 60))
        onset = float(get_note_value(note, "onset_beat", 0.0))
        duration = float(get_note_value(note, "duration_beat", 1.0))
        voice = int(get_note_value(note, "voice", 0))
        measure = measure_ids[i]

        note_probs = final_probs[i]

        predicted_labels = [
            LABEL_NAMES[j]
            for j, p in enumerate(note_probs)
            if float(p) >= threshold
        ]

        if not predicted_labels:
            predicted_labels = ["none"]

        top_index = int(torch.argmax(note_probs).item())
        top_label = LABEL_NAMES[top_index]

        rows.append(
            {
                "note_id": i,
                "measure": measure,
                "pitch": pitch,
                "note_name": midi_to_name(pitch),
                "onset_beat": onset,
                "duration_beat": duration,
                "voice": voice,
                "predicted_labels": "+".join(predicted_labels),
                "top_label": top_label,
                "scale_prob": float(note_probs[0]),
                "arpeggio_prob": float(note_probs[1]),
                "chord_prob": float(note_probs[2]),
                "jump_prob": float(note_probs[3]),
            }
        )

    return rows


def save_csv(rows: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_piano_roll(rows: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    label_colors = {
        "scale": "tab:blue",
        "arpeggio": "tab:green",
        "chord": "tab:purple",
        "jump": "tab:red",
        "none": "gray",
    }

    plt.figure(figsize=(16, 7))

    for row in rows:
        onset = row["onset_beat"]
        duration = row["duration_beat"]
        pitch = row["pitch"]
        
        if row["predicted_labels"] == "none":
            label = "none"
        else:
            label = row["top_label"]

        plt.hlines(
            y=pitch,
            xmin=onset,
            xmax=onset + duration,
            linewidth=6,
            color=label_colors.get(label, "gray"),
        )

        plt.text(onset, pitch + 0.25, str(row["note_id"]), fontsize=7)

    plt.xlabel("Onset beat")
    plt.ylabel("Pitch MIDI")
    plt.title("Windowed Predicted Piano Roll — GINE")

    legend_handles = [
        plt.Line2D([0], [0], color=color, lw=6, label=label)
        for label, color in label_colors.items()
    ]
    plt.legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("score", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--context", type=int, default=1)
    args = parser.parse_args()

    rows = predict_windowed(
        score_path=args.score,
        threshold=args.threshold,
        context=args.context,
    )

    base_name = args.score.stem
    csv_path = OUTPUT_DIR / f"{base_name}_windowed_gine_predictions.csv"
    png_path = OUTPUT_DIR / f"{base_name}_windowed_gine_piano_roll.png"

    save_csv(rows, csv_path)
    save_piano_roll(rows, png_path)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved piano roll: {png_path}")


if __name__ == "__main__":
    main()