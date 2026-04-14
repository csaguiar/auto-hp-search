"""
train.py — AGENT EDITS THIS FILE.

Rules of the game:
  * You may change anything below the "AGENT-EDITABLE REGION" banner:
    model architecture, optimizer, LR schedule, batch size,
    augmentation, regularization, training loop details, etc.
  * You MUST NOT modify prepare.py.
  * You MUST respect TIME_BUDGET_SECONDS of wall-clock training time
    (model construction + first warmup batch are excluded).
  * You MUST end the run by calling `log_run(...)` exactly once.

The metric is val_loss returned by prepare.evaluate(). Lower is better.
This baseline is deliberately mediocre — there is plenty of headroom.

Versioning: log_run() auto-commits the current train.py to git after each
run and stores the commit SHA in runs/log.jsonl. To recover any past
experiment's code, run:  git show <sha>:train.py
"""

from __future__ import annotations
import json
import time
import pathlib
import subprocess
import datetime as dt

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    NUM_CLASSES, IMG_SHAPE, get_loaders, evaluate,
)

# ---------------------------------------------------------------------------
# Fixed contract — do not change.
# ---------------------------------------------------------------------------
TIME_BUDGET_SECONDS = 90          # wall-clock training cap
RUNS_DIR            = pathlib.Path("runs")
RUNS_DIR.mkdir(exist_ok=True)


def _git(*args: str) -> str:
    """Run a git command; return stdout stripped, or '' on failure."""
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _commit_train_py(metrics: dict, notes: str) -> str:
    """Commit the current train.py and return the commit SHA (or '')."""
    if not _git("rev-parse", "--is-inside-work-tree"):
        return ""  # not a git repo — silently skip
    _git("add", "train.py")
    # `git diff --cached --quiet` exits 0 if nothing staged, 1 if changes.
    # We bypass the _git wrapper here so a non-zero exit isn't an error.
    has_changes = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        capture_output=True,
    ).returncode != 0
    if has_changes:
        msg = (f"exp {dt.datetime.now().isoformat(timespec='seconds')} | "
               f"val_loss={metrics.get('val_loss', float('nan')):.4f} | {notes}")
        _git("commit", "-m", msg)
    # If train.py was unchanged, HEAD still points at the commit that
    # produced this code — that's the right SHA to log.
    return _git("rev-parse", "HEAD")


def log_run(metrics: dict, notes: str = "") -> None:
    """Append one JSON line to runs/log.jsonl and commit train.py to git."""
    commit_sha = _commit_train_py(metrics, notes)
    record = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "commit":    commit_sha,
        "notes":     notes,
        **metrics,
    }
    with (RUNS_DIR / "log.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    print("LOGGED:", record)


# ===========================================================================
# >>>>>>>>>>>>>>>>>>>>>  AGENT-EDITABLE REGION BELOW  <<<<<<<<<<<<<<<<<<<<<<<
# ===========================================================================

BATCH_SIZE = 64
LR         = 1e-2
NOTES      = "baseline: 1-hidden-layer MLP, plain SGD, no schedule"


class Model(nn.Module):
    """Deliberately weak baseline: single hidden layer MLP."""
    def __init__(self):
        super().__init__()
        in_dim = IMG_SHAPE[0] * IMG_SHAPE[1] * IMG_SHAPE[2]
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Budget: {TIME_BUDGET_SECONDS}s")

    train_loader, _ = get_loaders(batch_size=BATCH_SIZE)
    model = Model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    # Warmup pass — excluded from the time budget so first-batch overhead
    # (cuDNN autotune, lazy init, etc.) doesn't penalize the agent.
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    F.cross_entropy(model(x), y).backward()
    optimizer.zero_grad()

    # ----- timed training loop -----
    # We log a smoothed train loss every CURVE_LOG_EVERY steps so the agent
    # can reason about training dynamics (plateau? still descending?
    # unstable?) rather than just the final number.
    CURVE_LOG_EVERY = 50
    curve = []                     # list of [step, smoothed_train_loss]
    ema = None                     # EMA of batch loss for a less noisy curve
    EMA_ALPHA = 0.1

    start = time.time()
    step = 0
    model.train()
    while True:
        for x, y in train_loader:
            if time.time() - start > TIME_BUDGET_SECONDS:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            step += 1

            loss_val = loss.item()
            ema = loss_val if ema is None else (1 - EMA_ALPHA) * ema + EMA_ALPHA * loss_val
            if step % CURVE_LOG_EVERY == 0:
                curve.append([step, round(ema, 4)])
            if step % 200 == 0:
                print(f"step {step:>5} | train_loss {loss_val:.4f} (ema {ema:.4f})")
        else:
            continue
        break

    elapsed = time.time() - start
    metrics = evaluate(model, device)
    metrics.update({
        "steps":          step,
        "elapsed_sec":    round(elapsed, 2),
        "batch_size":     BATCH_SIZE,
        "lr":             LR,
        "param_count":    sum(p.numel() for p in model.parameters()),
        "curve":          curve,
    })
    log_run(metrics, notes=NOTES)


if __name__ == "__main__":
    main()
