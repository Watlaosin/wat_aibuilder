from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv


BASE_DIR = Path(__file__).resolve().parent.parent

# for synthetic only or manual or mix. for benchmark and stuff
GRAPH_DIRS = [
    # synthetic only:
    # BASE_DIR / "dataset" / "synthetic_graphs",

    # manual only:
    # BASE_DIR / "dataset" / "graphs",

    # both:
    BASE_DIR / "dataset" / "synthetic_graphs",
    BASE_DIR / "dataset" / "graphs",
]

MODEL_PATH = BASE_DIR / "models" / "GINE_mixed_try.pt"

LABEL_NAMES = ["scale", "arpeggio", "chord", "jump"]

BATCH_SIZE = 32
EPOCHS = 50
LR = 0.001
WEIGHT_DECAY = 1e-4
SEED = 42

TRAIN_RATIO = 0.7


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


def load_graphs(graph_dirs: list[Path]):
    graphs = []

    for graph_dir in graph_dirs:
        graph_dir = Path(graph_dir)

        if not graph_dir.exists():
            raise FileNotFoundError(f"Graph directory does not exist: {graph_dir}")

        paths = sorted(graph_dir.rglob("*.pt"))

        if not paths:
            raise FileNotFoundError(f"No .pt graphs found in {graph_dir}")

        print(f"Found {len(paths)} graphs in {graph_dir}")

        for idx, path in enumerate(paths):
            data = torch.load(path, weights_only=False)

            if not hasattr(data, "target_mask"):
                raise ValueError(f"{path} is missing target_mask")

            if not hasattr(data, "y"):
                raise ValueError(f"{path} is missing y")

            if not hasattr(data, "piece_id"):
                # Fallback only. Your synthetic/manual graphs should already have piece_id.
                data.piece_id = path.parent.name

            if not hasattr(data, "source"):
                if "synthetic" in str(path).lower():
                    data.source = "synthetic"
                else:
                    data.source = "manual"

            data.graph_path = str(path)

            graphs.append(data)

    if not graphs:
        raise FileNotFoundError("No graphs were loaded.")

    return graphs


def check_graphs(graphs):
    x_dim = graphs[0].x.shape[1]
    edge_dim = graphs[0].edge_attr.shape[1]
    y_dim = graphs[0].y.shape[1]

    for i, graph in enumerate(graphs):
        if graph.x.shape[1] != x_dim:
            raise ValueError(
                f"Graph {i} has x dim {graph.x.shape[1]}, expected {x_dim}"
            )

        if graph.edge_attr.shape[1] != edge_dim:
            raise ValueError(
                f"Graph {i} has edge_attr dim {graph.edge_attr.shape[1]}, expected {edge_dim}"
            )

        if graph.y.shape[1] != y_dim:
            raise ValueError(
                f"Graph {i} has y dim {graph.y.shape[1]}, expected {y_dim}"
            )

        if graph.target_mask.shape[0] != graph.y.shape[0]:
            raise ValueError(
                f"Graph {i} target_mask length does not match y rows."
            )

    if y_dim != len(LABEL_NAMES):
        raise ValueError(
            f"y dim is {y_dim}, but LABEL_NAMES has {len(LABEL_NAMES)} labels."
        )

    print("Graph compatibility check passed.")
    print(f"x_dim={x_dim}, edge_dim={edge_dim}, y_dim={y_dim}")


def split_train_val_by_piece(graphs, train_ratio=0.7, seed=42):
    """
    Splits graphs by piece_id into train and validation only.

    Example:
        70% of pieces -> train
        30% of pieces -> validation

    No test set here. Test should be kept in a separate folder/script later.
    """
    piece_to_graphs = defaultdict(list)

    for graph in graphs:
        piece_id = str(graph.piece_id)
        piece_to_graphs[piece_id].append(graph)

    piece_ids = list(piece_to_graphs.keys())

    if len(piece_ids) < 2:
        raise ValueError(
            f"Need at least 2 piece_id values for train/val split. "
            f"Found only {len(piece_ids)}: {piece_ids}"
        )

    rng = random.Random(seed)
    rng.shuffle(piece_ids)

    n = len(piece_ids)
    train_end = int(n * train_ratio)

    train_piece_ids = set(piece_ids[:train_end])
    val_piece_ids = set(piece_ids[train_end:])

    if not train_piece_ids or not val_piece_ids:
        raise ValueError(
            "Train or validation split is empty. Add more pieces or adjust TRAIN_RATIO."
        )

    train_graphs = []
    val_graphs = []

    for piece_id, piece_graphs in piece_to_graphs.items():
        if piece_id in train_piece_ids:
            train_graphs.extend(piece_graphs)
        else:
            val_graphs.extend(piece_graphs)

    return (
        train_graphs,
        val_graphs,
        train_piece_ids,
        val_piece_ids,
    )


def strip_non_tensor_metadata(graphs):
    attrs_to_remove = [
        "piece_id",
        "source",
        "graph_path",
        "context_measures",
    ]

    for graph in graphs:
        for attr in attrs_to_remove:
            if hasattr(graph, attr):
                delattr(graph, attr)

    return graphs


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_labels = 0

    for data in loader:
        data = data.to(device)

        logits = model(data.x, data.edge_index, data.edge_attr)

        target_logits = logits[data.target_mask]
        target_y = data.y[data.target_mask]

        loss = criterion(target_logits, target_y)
        total_loss += loss.item()

        preds = (torch.sigmoid(target_logits) >= 0.5).float()

        total_correct += (preds == target_y).sum().item()
        total_labels += target_y.numel()

    avg_loss = total_loss / max(1, len(loader))
    label_accuracy = total_correct / max(1, total_labels)

    return avg_loss, label_accuracy


def count_sources(graphs):
    counts = defaultdict(int)

    for graph in graphs:
        source = getattr(graph, "source", "unknown")
        counts[source] += 1

    return dict(counts)


def count_target_labels(graphs):
    label_counts = torch.zeros(len(LABEL_NAMES))
    total_target_notes = 0
    none_count = 0

    for graph in graphs:
        y = graph.y[graph.target_mask]

        if y.numel() == 0:
            continue

        label_counts += y.sum(dim=0).cpu()
        total_target_notes += y.shape[0]
        none_count += int((y.sum(dim=1) == 0).sum().item())

    return label_counts, total_target_notes, none_count


def print_dataset_summary(name: str, graphs: list) -> None:
    print()
    print(f"{name}:")
    print(f"graphs: {len(graphs)}")
    print(f"sources: {count_sources(graphs)}")

    if not graphs:
        return

    label_counts, total_target_notes, none_count = count_target_labels(graphs)

    print("target-note label counts:")
    for label_name, count in zip(LABEL_NAMES, label_counts.tolist()):
        print(f"  {label_name}: {int(count)}")

    print(f"target notes: {total_target_notes}")
    print(f"none/all-zero target notes: {none_count}")


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graphs = load_graphs(GRAPH_DIRS)
    check_graphs(graphs)

    print()
    print(f"Loaded graphs: {len(graphs)}")
    print(f"Sources: {count_sources(graphs)}")

    label_counts, total_target_notes, none_count = count_target_labels(graphs)

    print()
    print("All target-note label counts:")
    for label_name, count in zip(LABEL_NAMES, label_counts.tolist()):
        print(f"{label_name}: {int(count)}")

    print(f"target notes: {total_target_notes}")
    print(f"none/all-zero target notes: {none_count}")

    (
        train_graphs,
        val_graphs,
        train_pieces,
        val_pieces,
    ) = split_train_val_by_piece(
        graphs,
        train_ratio=TRAIN_RATIO,
        seed=SEED,
    )

    print()
    print("Split mode: piece_id train/val only")
    print(f"Train ratio: {TRAIN_RATIO:.2f}")
    print(
        f"Train pieces: {len(train_pieces)} | "
        f"Val pieces: {len(val_pieces)}"
    )
    print(
        f"Train graphs: {len(train_graphs)} | "
        f"Val graphs: {len(val_graphs)}"
    )

    print_dataset_summary("Train set", train_graphs)
    print_dataset_summary("Validation set", val_graphs)

    train_graphs = strip_non_tensor_metadata(train_graphs)
    val_graphs = strip_non_tensor_metadata(val_graphs)

    in_channels = graphs[0].x.shape[1]
    edge_dim = graphs[0].edge_attr.shape[1]

    model = TechniqueGINE(
        in_channels=in_channels,
        edge_dim=edge_dim,
        out_channels=len(LABEL_NAMES),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for data in train_loader:
            data = data.to(device)

            optimizer.zero_grad()

            logits = model(data.x, data.edge_index, data.edge_attr)

            target_logits = logits[data.target_mask]
            target_y = data.y[data.target_mask]

            loss = criterion(target_logits, target_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / max(1, len(train_loader))
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_label_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    print()
    print("Training finished.")
    print("Final test skipped. Use a separate held-out test folder/script later.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Saved best model to: {MODEL_PATH}")


if __name__ == "__main__":
    train()