"""CLI entry point for the Duan et al. (MUDVI) baseline continual-learning
run on BCI IV 2a, subjects A01 -> A09.

Hyperparameters not given numerically by Duan et al. for BCI IV-2a
training (learning rate, batch size, epochs per subject) are NOT
specified in the paper beyond the memory-specific ones (memory size 200
by default per their ablation study, lambda=1). Defaults below are
reasonable choices, exposed as CLI flags, and clearly separate from the
paper-specified numbers (IMPLEMENTATION DECISION).

Usage:
  python -m mudvi_baseline.run_baseline --data_dir /path/to/BCICIV_2a_gdf
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .data import load_subject
from .model import MudviCNN
from .memory import ClassBalancedMemory
from .trainer import ContinualTrainer
from .metrics import compute_bwt

DEFAULT_SUBJECT_ORDER = [f"{i:02d}" for i in range(1, 10)]  # 01..09


def main():
    parser = argparse.ArgumentParser(description="MUDVI baseline: sequential continual EEG decoding on BCI IV 2a.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--subjects", type=str, default=",".join(DEFAULT_SUBJECT_ORDER),
                         help="Comma-separated subject order, e.g. 01,02,...,09")
    parser.add_argument("--memory_size", type=int, default=200,
                         help="Total replay buffer capacity (Duan et al. Table 3 default).")
    parser.add_argument("--epochs_per_subject", type=int, default=15)
    parser.add_argument("--new_batch_size", type=int, default=32)
    parser.add_argument("--mem_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gradient_projection", action="store_true",
                         help="Addition 1 (mudvi_gp_report.pdf Sec 3.2): use magnitude-aware "
                              "gradient projection between the memory and new-subject gradients "
                              "instead of the baseline combined-loss update. Default: off "
                              "(runs the original baseline unchanged), so results can be "
                              "compared against this same script run without the flag.")
    parser.add_argument("--relationship_shift_detection", action="store_true",
                         help="Addition 2 (mudvi_gp_report.pdf Sec 3.3): run a second, "
                              "independent shift detector alongside the existing one -- a "
                              "frozen reference model (trained on the first subject only) "
                              "tracks a NOT/GLR change-point signal over its confidence/loss "
                              "on incoming data, combined as final_shift = mmd_shift OR "
                              "confidence_shift. Purely diagnostic/logged, never gates "
                              "training. Default: off (baseline unchanged).")
    parser.add_argument("--out", type=str, default=None, help="Optional path to dump results JSON.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    subject_ids = args.subjects.split(",")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading and segmenting subjects: {subject_ids}")
    subjects_data = []
    for sid in subject_ids:
        sd = load_subject(sid, args.data_dir, test_fraction=args.test_fraction, seed=args.seed)
        print(f"  A{sid}T: train segments={len(sd.y_train)}, test segments={len(sd.y_test)}")
        subjects_data.append(sd)

    model = MudviCNN()
    memory = ClassBalancedMemory(capacity=args.memory_size, seed=args.seed)
    trainer = ContinualTrainer(
        model, memory, device,
        lr=args.lr,
        new_batch_size=args.new_batch_size,
        mem_batch_size=args.mem_batch_size,
        epochs_per_subject=args.epochs_per_subject,
        seed=args.seed,
        use_gradient_projection=args.gradient_projection,
        relationship_shift_detection=args.relationship_shift_detection,
    )
    mode_bits = []
    mode_bits.append("Gradient Projection (Addition 1)" if args.gradient_projection else None)
    mode_bits.append("Relationship-Shift Detection (Addition 2)" if args.relationship_shift_detection else None)
    mode_bits = [m for m in mode_bits if m]
    print(f"Mode: {' + '.join(mode_bits) if mode_bits else 'Baseline'}")

    acc_matrix = trainer.run(subjects_data)

    last_step = max(acc_matrix)
    print("\nAccuracy after each subject (Acc(i,i)) and after full sequence (Acc(N,i)):")
    print(f"{'Subject':<10}{'Acc(i,i)':<12}{'Acc(N,i)':<12}")
    for step_index, sd in enumerate(subjects_data):
        sid = sd.subject_id
        acc_ii = acc_matrix[step_index][sid]
        acc_Ni = acc_matrix[last_step][sid]
        print(f"A{sid:<9}{acc_ii:<12.4f}{acc_Ni:<12.4f}")

    bwt = compute_bwt(acc_matrix, subject_ids)
    print(f"\nBWT: {bwt:.4f}")
    print("Final memory class counts:", memory.class_counts())

    gp_summary = trainer.gp_stats.summary() if trainer.gp_stats is not None else None
    if gp_summary is not None:
        print("\nGradient projection stats (Addition 1):")
        for k, v in gp_summary.items():
            print(f"  {k}: {v}")

    shift_summary = None
    if args.relationship_shift_detection:
        log = trainer.shift_log
        total = len(log)
        shift_summary = {
            "total_steps_logged": total,
            "mmd_shifts": sum(1 for e in log if e["mmd_shift"]),
            "confidence_shifts": sum(1 for e in log if e["confidence_shift"]),
            "final_shifts": sum(1 for e in log if e["final_shift"]),
            "confidence_only_shifts": sum(1 for e in log if e["confidence_shift"] and not e["mmd_shift"]),
        }
        print("\nShift detection stats (Addition 2, mmd_shift OR confidence_shift):")
        for k, v in shift_summary.items():
            print(f"  {k}: {v}")

    if args.out:
        serializable = {str(k): v for k, v in acc_matrix.items()}
        results = {"acc_matrix": serializable, "bwt": bwt,
                   "memory_class_counts": memory.class_counts(),
                   "gradient_projection": args.gradient_projection,
                   "relationship_shift_detection": args.relationship_shift_detection}
        if gp_summary is not None:
            results["gradient_projection_stats"] = gp_summary
        if shift_summary is not None:
            results["shift_detection_stats"] = shift_summary
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
