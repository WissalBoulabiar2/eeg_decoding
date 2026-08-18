"""Checkpoint save/restore orchestration for remote (Lightning AI) runs.

Granularity: one checkpoint per completed SUBJECT (not per epoch/step
within a subject) -- Section 10/11 of the experimental brief asks for
runs to be "restartable where practical", and subject-boundary
granularity is the natural unit here: a MUDVI-style continual run is a
sequence of at most 9 subject-training phases, each taking minutes, so
losing at most one in-progress subject's training to a preemption is an
acceptable, well-scoped restart cost -- checkpointing every mini-batch
would add substantial I/O overhead for no practical benefit at this
scale.

A checkpoint captures everything `ContinualTrainer.state_dict()` returns
(model, optimizer, memory buffer contents, GP/RSD diagnostic state, RNG
state -- see that method's docstring for what is and is not covered) plus
the step index it was taken after and the experiment config it was run
with, so a resumed run can assert the config matches instead of silently
continuing with mismatched hyperparameters.
"""
from __future__ import annotations

import glob
import json
import os

import torch


def _checkpoint_dir(run_dir: str) -> str:
    d = os.path.join(run_dir, "checkpoints")
    os.makedirs(d, exist_ok=True)
    return d


def save_checkpoint(trainer, run_dir: str, step_index: int, config_dict: dict) -> str:
    ckpt_dir = _checkpoint_dir(run_dir)
    path = os.path.join(ckpt_dir, f"step_{step_index:02d}.pt")
    payload = {
        "step_index": step_index,
        "config": config_dict,
        "trainer_state": trainer.state_dict(),
    }
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)  # atomic on POSIX and Windows NTFS: no half-written checkpoint on crash
    return path


def find_latest_checkpoint(run_dir: str):
    """Returns (step_index, path) for the highest-step_index checkpoint
    found in `run_dir/checkpoints/`, or None if none exist."""
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None
    candidates = glob.glob(os.path.join(ckpt_dir, "step_*.pt"))
    if not candidates:
        return None
    best_step, best_path = -1, None
    for path in candidates:
        base = os.path.basename(path)
        try:
            step = int(base[len("step_"):-len(".pt")])
        except ValueError:
            continue
        if step > best_step:
            best_step, best_path = step, path
    if best_path is None:
        return None
    return best_step, best_path


def load_checkpoint(path: str, map_location=None) -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def assert_config_matches(checkpoint_config: dict, current_config: dict, ignore_keys=("resume", "checkpoint_every")) -> None:
    """Refuses to resume a run with a checkpoint taken under different
    hyperparameters -- a silent hyperparameter drift across a
    preemption/restart would invalidate the comparison without anyone
    noticing (Section 9's fairness requirement applies across restarts of
    the SAME run too, not just across different methods)."""
    mismatches = []
    for k in checkpoint_config:
        if k in ignore_keys:
            continue
        if checkpoint_config.get(k) != current_config.get(k):
            mismatches.append((k, checkpoint_config.get(k), current_config.get(k)))
    if mismatches:
        lines = "\n".join(f"  {k}: checkpoint={a!r} vs. current={b!r}" for k, a, b in mismatches)
        raise ValueError(
            f"Refusing to resume: checkpoint config differs from the current run's config:\n{lines}\n"
            f"Use a different --run_name/--out_dir for a genuinely new run instead of --resume."
        )


def save_metrics(run_dir: str, metrics: dict) -> str:
    path = os.path.join(run_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path


def save_config(run_dir: str, config_dict: dict) -> str:
    path = os.path.join(run_dir, "config.json")
    with open(path, "w") as f:
        json.dump(config_dict, f, indent=2)
    return path
