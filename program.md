# program.md — Autoresearch Agent Instructions

You are an autonomous ML research agent. Your job is to iteratively improve
the validation loss on Fashion-MNIST by editing **one file**: `train.py`.

## The setup

* **`prepare.py`** — frozen. Provides `get_loaders()` and `evaluate(model)`.
  Do **not** read or modify it beyond importing from it. Do not change the
  validation split, the metric, or the data normalization.
* **`train.py`** — your single editable surface. The region below the
  `AGENT-EDITABLE REGION` banner is yours. Everything above it (the
  `log_run` helper, the time budget constant, the imports) is fixed.
* **`pyproject.toml`** — dependency manifest, managed by `uv`. You may add
  packages with `uv add <pkg>` if an experiment genuinely needs one
  (e.g., `timm`), but prefer pure-PyTorch solutions.
* **`runs/log.jsonl`** — append-only history of every experiment. One JSON
  line per run, including a git commit SHA pointing at the exact
  `train.py` that produced it.

## One-time setup

```
# 1. Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create the virtualenv and install dependencies
uv sync

# 3. Initialize git so the harness can version each experiment
git init
git add prepare.py train.py program.md pyproject.toml uv.lock
git commit -m "initial baseline"

# 4. One-time data download + harness smoke test
uv run python prepare.py
```

To recover the code from any past run:

```
git show <sha>:train.py
```

## The metric

Lower **`val_loss`** (cross-entropy on the held-out 5,000-image validation
split) is better. `val_acc` is informational only — do not optimize it
directly. The metric is comparable across architectures because the
val split, normalization, and evaluation code are frozen.

## The learning curve

Every log record also contains a `curve` field: a list of
`[step, smoothed_train_loss]` pairs sampled every 50 steps. Do **not**
compare runs using `curve` — use `val_loss` for that. Instead, read
`curve` to diagnose *why* a run got the number it got, and use that
diagnosis to choose the next change. A few patterns worth recognizing:

* **Still descending at the end** → the run was budget-limited, not
  capacity-limited. Try a higher learning rate, a larger batch, or a
  more efficient architecture so each step learns more.
* **Flat plateau for the last third** → converged; further training
  wouldn't help. Improvement has to come from architecture, optimizer,
  or regularization.
* **Descended then climbed back up** → overfitting or LR too high late
  in training. Add a schedule (cosine decay), regularization (dropout,
  weight decay), or augmentation.
* **Spikes / NaN-ish jumps** → instability. Lower LR, add gradient
  clipping, or switch to a more stable optimizer.
* **Noisy but trending down** → normal; curve smoothing is an EMA so
  some noise survives. Don't over-interpret.

Note: the curve is train loss only, not val loss (we don't want
mid-training evals to eat into the time budget). Overfitting shows up
indirectly — a final `val_loss` much worse than the curve's tail is the
tell.

## The contract for every experiment

1. **Read `runs/log.jsonl`** to see what has already been tried and the
   current best `val_loss`.
2. **Propose ONE focused change** to `train.py`. Examples of valid changes:
   architecture (depth, width, conv vs MLP, normalization, activations),
   optimizer (Adam, AdamW, SGD+momentum), learning rate + schedule
   (warmup, cosine, step decay), batch size, weight decay, dropout, data
   augmentation (random crops, flips — note: horizontal flip is reasonable
   for clothing), label smoothing, mixed precision.
3. **Update the `NOTES` string** in `train.py` to a one-line description of
   what you changed and why. This becomes part of the commit message.
4. **Run** `uv run python train.py`. It will train for `TIME_BUDGET_SECONDS`,
   evaluate, append a record to `runs/log.jsonl`, and auto-commit
   `train.py` to git.
5. **Compare** the new `val_loss` to the previous best.
   * If **better** → keep the edit. The new HEAD is your baseline.
   * If **worse or equal** → revert with
     `git checkout HEAD~1 -- train.py` (or check out the SHA of the
     current best from the log) before the next iteration.
6. **Repeat.**

## Rules

* Always invoke Python through `uv run` so the pinned environment is used.
  Never `pip install` directly — use `uv add <pkg>` if you need a new
  dependency, then commit `pyproject.toml` and `uv.lock`.
* Only `train.py` may be edited (plus `pyproject.toml`/`uv.lock` if you
  add a dependency). Do not touch `prepare.py`, do not add new files that
  bypass the harness, do not change `TIME_BUDGET_SECONDS`, and do not call
  `evaluate()` more than the one time the harness already does.
* No peeking at the test set. There is no test set in this PoC by design.
* One change per experiment. Do not bundle five tweaks into one diff —
  you will not learn which one helped, and the git history becomes useless.
* Stay within the time budget. If your model is too large to make
  meaningful progress in 60 seconds, that is itself a negative result —
  log it and move on.

## Suggested first moves (in roughly increasing ambition)

1. Switch SGD → AdamW with `lr=3e-4`, `weight_decay=1e-4`.
2. Replace the MLP with a small CNN (2 conv blocks → global pool → linear).
3. Add a cosine LR schedule with a short linear warmup.
4. Add light augmentation: random crop with padding=2, horizontal flip.
5. Add BatchNorm or GroupNorm after each conv.
6. Sweep batch size {64, 128, 256} once a good architecture is settled.

## How to start

After the one-time setup above:

```
uv run python train.py    # baseline run, ~60s
```

Then begin the loop above. Good luck.
