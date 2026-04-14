"""
prepare.py — FROZEN. Do not modify.

Provides:
  * One-time Fashion-MNIST download + deterministic train/val split.
  * get_loaders(batch_size) -> (train_loader, val_loader)
  * evaluate(model, device) -> {"val_loss": float, "val_acc": float}

The val set is a fixed 5,000-image slice of the official train set
(seed=0). The official test set is held out and NOT used here, so the
agent cannot overfit to it during the autoresearch loop.

Primary metric: val_loss (cross-entropy, lower is better).
Secondary:     val_acc  (informational only).
"""

from __future__ import annotations
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Fixed constants — the agent must not change these.
# ---------------------------------------------------------------------------
DATA_DIR        = os.environ.get("DATA_DIR", "./data")
NUM_CLASSES     = 10
IMG_SHAPE       = (1, 28, 28)            # (C, H, W)
VAL_SIZE        = 5_000                  # held out from the 60k train set
SPLIT_SEED      = 0
EVAL_BATCH_SIZE = 512


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------
_NORMALIZE = transforms.Compose([
    transforms.ToTensor(),                          # -> [0, 1]
    transforms.Normalize((0.2860,), (0.3530,)),     # FashionMNIST stats
])


def _build_splits():
    """Download (if needed) and produce deterministic train/val Subsets."""
    full = datasets.FashionMNIST(
        root=DATA_DIR, train=True, download=True, transform=_NORMALIZE,
    )
    g = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(len(full), generator=g).tolist()
    val_idx, train_idx = perm[:VAL_SIZE], perm[VAL_SIZE:]
    return Subset(full, train_idx), Subset(full, val_idx)


def get_loaders(batch_size: int):
    """Return (train_loader, val_loader). The agent picks batch_size."""
    train_set, val_set = _build_splits()
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Evaluation — the single source of truth for the metric.
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: torch.nn.Module, device: torch.device) -> dict:
    """Return {'val_loss', 'val_acc'} on the fixed validation split."""
    _, val_loader = get_loaders(batch_size=EVAL_BATCH_SIZE)
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for x, y in val_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        total_loss    += F.cross_entropy(logits, y, reduction="sum").item()
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_n       += y.size(0)
    return {
        "val_loss": total_loss / total_n,
        "val_acc":  total_correct / total_n,
    }


if __name__ == "__main__":
    # Smoke test: download + one eval pass with an untrained linear model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(28 * 28, NUM_CLASSES),
    ).to(device)
    print("Device:", device)
    print("Eval (untrained):", evaluate(dummy, device))
