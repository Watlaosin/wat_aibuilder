from __future__ import annotations

from pathlib import Path
import sys
import torch

from parser import load_score, extract_notes
from graph_builder import build_graph

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(BASE_DIR / "test_and_debug"))
from visualize_graph import visualize_music_graph

RAW_DIR = BASE_DIR / "dataset" / "raw"
GRAPH_DIR = BASE_DIR / "dataset" / "graphs"
PREVIEW_DIR = BASE_DIR / "dataset" / "previews"


def process_file(file_path: Path, graph_save_path: Path, preview_save_path: Path) -> None:
    print(f"Processing: {file_path}")

    score = load_score(str(file_path))
    notes = extract_notes(score)

    if len(notes) == 0:
        print(f"Skipped empty: {file_path}")
        return

    data = build_graph(notes)

    num_nodes = data.x.shape[0]
    data.y = torch.zeros((num_nodes, 4), dtype=torch.float32)

    torch.save(data, graph_save_path)
    visualize_music_graph(data, str(preview_save_path))

    print(f"Saved graph:   {graph_save_path}")
    print(f"Saved preview: {preview_save_path}")


def main() -> None:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    for file_path in RAW_DIR.iterdir():
        if file_path.suffix.lower() not in {".xml", ".mxl"}:
            continue

        base_name = file_path.stem
        graph_output_path = GRAPH_DIR / f"{base_name}.pt"
        preview_output_path = PREVIEW_DIR / f"{base_name}.png"

        process_file(file_path, graph_output_path, preview_output_path)


if __name__ == "__main__":
    main()