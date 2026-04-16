# auto-hp-search

A minimal harness for running **autoresearch loops** — an LLM agent iteratively
editing a training script, running it, reading the results, and deciding what
to try next. The ML task (a small Fashion-MNIST classifier) is a toy stand-in;
the point of this repo is the research *setup*, not the model.

This project is a streamlined adaptation of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) to
work on a simpler deep learning example.

## What's interesting here

Most "agent writes ML code" demos let the agent run wild: rewrite the data
loader, change the metric, evaluate on whatever it wants. That gives you
results you can't trust and a git history you can't learn from. This harness
constrains the loop so each iteration produces a comparable, reproducible data
point.

The key design choices:

- **One editable surface.** The agent only touches `train.py` below an
  `AGENT-EDITABLE REGION` banner. The data split, normalization, metric, and
  logging are in `prepare.py` and the top of `train.py`, both frozen. This
  means every run's `val_loss` is comparable across architectures, optimizers,
  and hyperparameters — the only thing that varies is what the agent changed.
- **Fixed wall-clock budget.** Every run trains for `TIME_BUDGET_SECONDS` of
  wall time (model construction and one warmup batch are excluded so first-
  batch overhead doesn't penalize the agent). This makes runs cheap and turns
  "your model is too big to train in 90s" into a signal the agent can learn
  from rather than a crash.
- **No test set.** The held-out split is a 5,000-image validation slice of the
  official train set, seeded deterministically. The official test set is never
  touched in the loop, so the agent can't overfit to it across iterations.
- **One change per experiment.** The agent is instructed to make a single
  focused edit per run. Bundled changes destroy the ability to attribute
  improvement to a cause, and make the git log useless as a record of what
  works.
- **Git is the experiment store.** After each run, `log_run()` auto-commits
  `train.py` and records the commit SHA alongside the metrics in
  `runs/log.jsonl`. Reverting a regression is `git checkout HEAD~1 -- train.py`;
  recovering any past experiment's exact code is `git show <sha>:train.py`.
  No separate "experiment tracker" dependency.
- **Learning curves, not just final numbers.** Each log record includes a
  `curve` field — `[step, smoothed_train_loss]` sampled every 50 steps. The
  agent is told to use `val_loss` for ranking runs and the curve for
  *diagnosing* them (still descending → budget-limited; plateau → converged;
  climbs back up → overfit; spikes → unstable). This pushes the agent toward
  root-cause reasoning instead of random search.

## Repo layout

```
program.md         Instructions the agent reads at the start of the loop.
prepare.py         Frozen. Data loading + evaluate(). Defines the metric.
train.py           The agent's editable surface. Top half is fixed harness.
pyproject.toml     uv-managed dependencies.
runs/log.jsonl     Append-only experiment log (created on first run).
data/              Fashion-MNIST cache (downloaded on first run).
```

## The loop

`program.md` is the system prompt. In short, each iteration:

1. Read `runs/log.jsonl` to see what's been tried and the current best.
2. Propose one focused change to `train.py`, updating the `NOTES` string.
3. `uv run python train.py` — trains, evaluates, appends to the log,
   auto-commits.
4. Compare `val_loss` to the previous best. Keep the edit if better, revert
   with `git checkout HEAD~1 -- train.py` if not.
5. Repeat.

## Running the harness yourself

```bash
# One-time
uv sync
git init && git add -A && git commit -m "initial baseline"
uv run python prepare.py         # downloads data, smoke-tests evaluate()

# One run
uv run python train.py           # trains for TIME_BUDGET_SECONDS, logs a row
```

Point an agent (Claude Code, or anything else that can edit files and run
shell commands) at `program.md` and let it iterate. The model and dataset are
deliberately small so a loop of dozens of iterations is cheap.

## Adapting it to a real problem

The toy pieces to swap out: the dataset and `IMG_SHAPE` in `prepare.py`, the
baseline `Model` in `train.py`, and `TIME_BUDGET_SECONDS`. The parts worth
keeping as-is: the frozen-vs-editable split, the single-file edit surface, the
auto-commit-per-run logging, and the curve-level feedback to the agent. Those
are what make the results interpretable.
