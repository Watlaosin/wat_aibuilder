from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader


BASE_DIR = Path(__file__).resolve().parent.parent
GRAPH_DIR = BASE_DIR / "dataset" / "synthetic_graphs"
MODEL_PATH = BASE_DIR / "best_synthetic_gine.pt"

LABEL_NAMES = ["scale", "arpeggio", "chord", "jump"]


from torch_geometric.nn import GINEConv


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


def load_graphs(graph_dir: Path, measures_per_piece: int = 8):
    paths = sorted(graph_dir.glob("*.pt"))

    if not paths:
        raise FileNotFoundError(f"No .pt graphs found in {graph_dir}")

    graphs = []

    for idx, path in enumerate(paths):
        data = torch.load(path, weights_only=False)

        if not hasattr(data, "target_mask"):
            raise ValueError(f"{path} is missing target_mask")

        if not hasattr(data, "piece_id"):
            data.piece_id = idx // measures_per_piece

        graphs.append(data)

    return graphs


def split_by_piece(graphs, train_ratio=0.8, val_ratio=0.1, seed=42):
    piece_to_graphs = defaultdict(list)

    for graph in graphs:
        piece_to_graphs[int(graph.piece_id)].append(graph)

    piece_ids = list(piece_to_graphs.keys())

    rng = random.Random(seed)
    rng.shuffle(piece_ids)

    n = len(piece_ids)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_piece_ids = set(piece_ids[:train_end])
    val_piece_ids = set(piece_ids[train_end:val_end])
    test_piece_ids = set(piece_ids[val_end:])

    train_graphs = []
    val_graphs = []
    test_graphs = []

    for piece_id, piece_graphs in piece_to_graphs.items():
        if piece_id in train_piece_ids:
            train_graphs.extend(piece_graphs)
        elif piece_id in val_piece_ids:
            val_graphs.extend(piece_graphs)
        else:
            test_graphs.extend(piece_graphs)

    return train_graphs, val_graphs, test_graphs, train_piece_ids, val_piece_ids, test_piece_ids


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


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graphs = load_graphs(GRAPH_DIR, measures_per_piece=8)
    train_graphs, val_graphs, test_graphs, train_pieces, val_pieces, test_pieces = split_by_piece(graphs)

    print(f"Loaded graphs: {len(graphs)}")
    print(f"Train pieces: {len(train_pieces)} | Val pieces: {len(val_pieces)} | Test pieces: {len(test_pieces)}")
    print(f"Train graphs: {len(train_graphs)} | Val graphs: {len(val_graphs)} | Test graphs: {len(test_graphs)}")

    in_channels = graphs[0].x.shape[1]
    edge_dim = graphs[0].edge_attr.shape[1]

    model = TechniqueGINE(
        in_channels=in_channels,
        edge_dim=edge_dim,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32)
    test_loader = DataLoader(test_graphs, batch_size=32)

    best_val_loss = float("inf")

    for epoch in range(1, 51):
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
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print("\nFinal test:")
    print(f"test_loss={test_loss:.4f}")
    print(f"test_label_acc={test_acc:.4f}")
    print(f"Saved best model to: {MODEL_PATH}")


if __name__ == "__main__":
    train()