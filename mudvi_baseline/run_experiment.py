"""Unified experiment launcher for remote (Lightning AI) execution.

Replaces manually editing source / juggling `run_baseline.py` vs.
`run_comparison.py` with one CLI surface covering every method this
project defines: mudvi (+GP/+RSD/+OCAR, any combination), er, mir, gmed
-- see `config.IMPLEMENTED_METHODS`.

Usage:
  python -m mudvi_baseline.run_experiment --method mudvi \\
      --gradient_projection --subjects 01,02,03,04,05,06,07,08,09 \\
      --memory_size 200 --seed 0 --out_dir results

  # resume a preempted run (same flags, plus --resume):
  python -m mudvi_baseline.run_experiment --method mudvi --gradient_projection \\
      --subjects 01,02,03,04,05,06,07,08,09 --memory_size 200 --seed 0 \\
      --out_dir results --resume

Results land in: {out_dir}/{result_subdir}/{run_id}/
  config.json          -- full resolved config, for reproducibility
  metrics.json          -- acc_matrix, BWT, forgetting, memory counts,
                            GP/RSD diagnostic summaries
  checkpoints/step_NN.pt -- one checkpoint per completed subject
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

from .config import parse_config, ExperimentConfig
from .data import load_subject as load_subject_bci2a
from .data_high_gamma import load_subject as load_subject_high_gamma
from .model import MudviCNN
from .memory import ClassBalancedMemory
from .er_baseline import ReservoirMemory
from .trainer import ContinualTrainer
from .mir_baseline import MIRTrainer
from .gmed_baseline import GMEDTrainer
from .metrics import compute_bwt, compute_forgetting
from . import checkpoint as ckpt


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_trainer(cfg: ExperimentConfig, model, device) -> ContinualTrainer:
    """Method dispatch. Each branch uses the SAME model class, optimizer
    (Adam, lr=cfg.lr), batch sizes, epoch count, and memory budget --
    the only thing that differs between methods is memory policy
    (class-balanced vs. global reservoir) and retrieval strategy (MIR's
    interference ranking vs. uniform sampling), per the fairness
    protocol (Section 9)."""
    if cfg.method == "mudvi":
        memory = ClassBalancedMemory(capacity=cfg.memory_size, seed=cfg.seed)
        return ContinualTrainer(
            model, memory, device, lr=cfg.lr,
            new_batch_size=cfg.new_batch_size, mem_batch_size=cfg.mem_batch_size,
            epochs_per_subject=cfg.epochs_per_subject, seed=cfg.seed,
            use_gradient_projection=cfg.gradient_projection,
            relationship_shift_detection=cfg.relationship_shift_detection,
            confidence_signal_type=cfg.confidence_signal_type,
            confidence_window_size=cfg.confidence_window_size,
            confidence_min_segment_length=cfg.confidence_min_segment_length,
            use_ocar=cfg.ocar,
            ocar_alpha_ema=cfg.ocar_alpha_ema,
            ocar_regul=cfg.ocar_regul,
            ocar_fim_update_every=cfg.ocar_fim_update_every,
            use_ocarpp=cfg.ocar_plusplus,
            ocarpp_beta_anchor=cfg.ocarpp_beta_anchor,
            ocarpp_gamma=cfg.ocarpp_gamma,
            ocarpp_eps=cfg.ocarpp_eps,
        )
    if cfg.method == "er":
        memory = ReservoirMemory(capacity=cfg.memory_size, seed=cfg.seed)
        return ContinualTrainer(
            model, memory, device, lr=cfg.lr,
            new_batch_size=cfg.new_batch_size, mem_batch_size=cfg.mem_batch_size,
            epochs_per_subject=cfg.epochs_per_subject, seed=cfg.seed,
            use_gradient_projection=False, relationship_shift_detection=False,
        )
    if cfg.method == "mir":
        memory = ReservoirMemory(capacity=cfg.memory_size, seed=cfg.seed)
        return MIRTrainer(
            model, memory, device, lr=cfg.lr,
            new_batch_size=cfg.new_batch_size, mem_batch_size=cfg.mem_batch_size,
            epochs_per_subject=cfg.epochs_per_subject, seed=cfg.seed,
            use_gradient_projection=False, relationship_shift_detection=False,
        )
    if cfg.method == "gmed":
        memory = ReservoirMemory(capacity=cfg.memory_size, seed=cfg.seed)
        return GMEDTrainer(
            model, memory, device, lr=cfg.lr,
            new_batch_size=cfg.new_batch_size, mem_batch_size=cfg.mem_batch_size,
            epochs_per_subject=cfg.epochs_per_subject, seed=cfg.seed,
            use_gradient_projection=False, relationship_shift_detection=False,
            edit_lr=cfg.gmed_edit_lr,
        )
    raise ValueError(f"Unknown method: {cfg.method!r}")


def run(cfg: ExperimentConfig) -> dict:
    """Runs one fully-resolved experiment config end to end and returns its
    metrics dict. Split out from `main()` so sweep drivers
    (`run_sensitivity.py`, `run_subject_order.py`) can build many
    `ExperimentConfig`s programmatically (e.g. via `dataclasses.replace`)
    and call this directly, instead of round-tripping each variant through
    CLI argv strings."""
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset is namespaced into the results path only for non-default
    # datasets, so BCI IV-2a's existing results/<result_subdir>/<run_id>/
    # layout (documented in LIGHTNING_MIGRATION.md) stays unchanged.
    dataset_parts = [] if cfg.dataset == "bci2a" else [cfg.dataset]
    run_dir = os.path.join(cfg.out_dir, *dataset_parts, cfg.result_subdir(), cfg.run_id())
    existing_ckpt = ckpt.find_latest_checkpoint(run_dir)
    if existing_ckpt is not None and not cfg.resume:
        raise SystemExit(
            f"Refusing to overwrite existing run at {run_dir} "
            f"(found checkpoint at step {existing_ckpt[0]}). Pass --resume to "
            f"continue it, or --run_name/--out_dir to start a distinct run."
        )
    os.makedirs(run_dir, exist_ok=True)

    print(f"[run_experiment] method={cfg.method} subjects={cfg.subjects} "
          f"gp={cfg.gradient_projection} rsd={cfg.relationship_shift_detection} "
          f"ocar={cfg.ocar} ocar_plusplus={cfg.ocar_plusplus} seed={cfg.seed} device={device}")
    print(f"[run_experiment] run_dir={run_dir}")

    load_subject = load_subject_bci2a if cfg.dataset == "bci2a" else load_subject_high_gamma

    print(f"[run_experiment] dataset={cfg.dataset} Loading and segmenting subjects: {cfg.subjects}")
    subjects_data = []
    for sid in cfg.subjects:
        sd = load_subject(sid, cfg.data_dir, test_fraction=cfg.test_fraction, seed=cfg.seed)
        print(f"  subject {sid}: train segments={len(sd.y_train)}, test segments={len(sd.y_test)}")
        subjects_data.append(sd)

    num_channels = subjects_data[0].X_train.shape[1]
    model = MudviCNN(num_channels=num_channels)
    trainer = build_trainer(cfg, model, device)

    start_step = 0
    if existing_ckpt is not None:
        step_index, path = existing_ckpt
        print(f"[run_experiment] Resuming from checkpoint step {step_index} ({path})")
        payload = ckpt.load_checkpoint(path, map_location=device)
        ckpt.assert_config_matches(payload["config"], cfg.to_dict())
        trainer.load_state_dict(payload["trainer_state"])
        start_step = payload["step_index"] + 1
        if start_step >= len(subjects_data):
            print("[run_experiment] Checkpoint already covers all requested subjects; nothing to do.")

    ckpt.save_config(run_dir, cfg.to_dict())

    def on_step_done(step_index: int):
        is_last = step_index == len(subjects_data) - 1
        should_checkpoint = cfg.checkpoint_every > 0 and (
            is_last or (step_index + 1) % cfg.checkpoint_every == 0
        )
        if should_checkpoint:
            path = ckpt.save_checkpoint(trainer, run_dir, step_index, cfg.to_dict())
            print(f"[run_experiment] checkpoint saved: {path}")

    t0 = time.time()
    acc_matrix = trainer.run(subjects_data, start_step=start_step, on_step_done=on_step_done)
    elapsed = time.time() - t0

    last_step = max(acc_matrix)
    print("\nAccuracy after each subject (Acc(i,i)) and after full sequence (Acc(N,i)):")
    print(f"{'Subject':<10}{'Acc(i,i)':<12}{'Acc(N,i)':<12}")
    for step_index, sd in enumerate(subjects_data):
        sid = sd.subject_id
        acc_ii = acc_matrix[step_index][sid]
        acc_Ni = acc_matrix[last_step][sid]
        print(f"A{sid:<9}{acc_ii:<12.4f}{acc_Ni:<12.4f}")

    bwt = compute_bwt(acc_matrix, cfg.subjects)
    forgetting = compute_forgetting(acc_matrix, cfg.subjects)
    final_avg_acc = float(np.mean([acc_matrix[last_step][sid] for sid in cfg.subjects]))
    print(f"\nBWT: {bwt:.4f}  Forgetting: {forgetting:.4f}  FinalAvgAcc: {final_avg_acc:.4f}")
    print(f"Elapsed: {elapsed:.1f}s  Final memory class counts: {trainer.memory.class_counts()}")

    metrics = {
        "method": cfg.method,
        "subjects": cfg.subjects,
        "seed": cfg.seed,
        "acc_matrix": {str(k): v for k, v in acc_matrix.items()},
        "bwt": bwt,
        "forgetting": forgetting,
        "final_avg_acc": final_avg_acc,
        "memory_class_counts": trainer.memory.class_counts(),
        "gradient_projection": cfg.gradient_projection,
        "relationship_shift_detection": cfg.relationship_shift_detection,
        "ocar": cfg.ocar,
        "ocar_plusplus": cfg.ocar_plusplus,
        "elapsed_seconds": elapsed,
    }
    if trainer.gp_stats is not None:
        gp_summary = trainer.gp_stats.summary()
        metrics["gradient_projection_stats"] = gp_summary
        print("\nGradient projection stats:")
        for k, v in gp_summary.items():
            print(f"  {k}: {v}")
    if trainer.ocar_stats is not None:
        ocar_summary = trainer.ocar_stats.summary()
        metrics["ocar_stats"] = ocar_summary
        print("\nOCAR stats:")
        for k, v in ocar_summary.items():
            print(f"  {k}: {v}")
    if trainer.ocarpp_stats is not None:
        ocarpp_summary = trainer.ocarpp_stats.summary()
        metrics["ocarpp_stats"] = ocarpp_summary
        print("\nOCAR++ stats:")
        for k, v in ocarpp_summary.items():
            print(f"  {k}: {v}")
    if cfg.relationship_shift_detection:
        log = trainer.shift_log
        shift_summary = {
            "total_steps_logged": len(log),
            "mmd_shifts": sum(1 for e in log if e["mmd_shift"]),
            "confidence_shifts": sum(1 for e in log if e["confidence_shift"]),
            "final_shifts": sum(1 for e in log if e["final_shift"]),
            "confidence_only_shifts": sum(1 for e in log if e["confidence_shift"] and not e["mmd_shift"]),
        }
        metrics["shift_detection_stats"] = shift_summary
        metrics["shift_log"] = log
        print("\nShift detection stats:")
        for k, v in shift_summary.items():
            print(f"  {k}: {v}")

    metrics_path = ckpt.save_metrics(run_dir, metrics)
    print(f"\n[run_experiment] metrics written to {metrics_path}")
    return metrics


def main(argv=None):
    cfg = parse_config(argv)
    return run(cfg)


if __name__ == "__main__":
    main()
