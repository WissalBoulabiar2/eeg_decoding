"""Six-way comparison for Prof. Duan's feedback: MUDVI / +GP / +RSD /
+GP+RSD / ER / MIR, on subjects A01-A03 ONLY.

COMPUTATIONAL SAFETY: `SUBJECTS` below is a hardcoded constant, not a CLI
flag -- there is no way to make this script touch A04-A09. The subject
list is printed before any data loading and again in the final summary.

Protocol: identical hyperparameters across all six methods
(memory_size=200, epochs_per_subject=3, new_batch_size=32,
mem_batch_size=32, lr=1e-3, test_fraction=0.2, seed=0), copied verbatim
from `run_stage1.sh`'s `COMMON` args -- the exact protocol already used
for the paper's "Stage 1" A01-A03 ablation
(mudvi_baseline/stage1_*.json). Per that existing, deterministic,
fixed-seed protocol, the four MUDVI-variant results are REUSED from those
JSON files rather than retrained (same code path, same seed -> same
numbers; use --rerun_mudvi to force retraining them instead). Only ER and
MIR are trained fresh here, strictly SEQUENTIALLY (never concurrently).

Usage:
  Smoke test (fast, tiny epochs/memory, sanity-checks the full pipeline):
    python -m mudvi_baseline.run_comparison --data_dir <dir> --smoke

  Real run (matches the Stage-1 protocol, ER + MIR only trained fresh):
    python -m mudvi_baseline.run_comparison --data_dir <dir>
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import load_subject
from .model import MudviCNN
from .memory import ClassBalancedMemory
from .trainer import ContinualTrainer
from .metrics import compute_bwt, compute_forgetting
from .er_baseline import ReservoirMemory
from .mir_baseline import MIRTrainer

SUBJECTS = ["01", "02", "03"]  # HARD-CODED. Never A04-A09. See module docstring.
SEED = 0

METHOD_ORDER = ["MUDVI", "MUDVI+GP", "MUDVI+RSD", "MUDVI+GP+RSD", "ER", "MIR"]
MUDVI_VARIANTS = ["MUDVI", "MUDVI+GP", "MUDVI+RSD", "MUDVI+GP+RSD"]
STAGE1_FILES = {
    "MUDVI": "stage1_baseline.json",
    "MUDVI+GP": "stage1_gp.json",
    "MUDVI+RSD": "stage1_rsd.json",
    "MUDVI+GP+RSD": "stage1_gp_rsd.json",
}
COLORS = {
    "MUDVI": "#1f77b4", "MUDVI+GP": "#ff7f0e", "MUDVI+RSD": "#2ca02c",
    "MUDVI+GP+RSD": "#d62728", "ER": "#9467bd", "MIR": "#8c564b",
}


def set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def summarize(method: str, acc_matrix: dict, memory_class_counts: dict,
              gp_stats=None, shift_stats=None) -> dict:
    """acc_matrix: {int_step: {subject_id: acc}}. Builds the unified
    per-method record used for the comparison table and figures."""
    last = max(acc_matrix)
    acc_ii = {sid: acc_matrix[step][sid] for step, sid in enumerate(SUBJECTS)}
    acc_Ni = {sid: acc_matrix[last][sid] for sid in SUBJECTS}
    return {
        "method": method,
        "acc_ii": acc_ii,
        "acc_Ni": acc_Ni,
        "final_avg_acc": float(np.mean([acc_Ni[sid] for sid in SUBJECTS])),
        "bwt": compute_bwt(acc_matrix, SUBJECTS),
        "forgetting": compute_forgetting(acc_matrix, SUBJECTS),
        "memory_class_counts": memory_class_counts,
        "gp_stats": gp_stats,
        "shift_stats": shift_stats,
    }


def load_stage1_result(stage1_dir: str, method: str) -> dict:
    path = os.path.join(stage1_dir, STAGE1_FILES[method])
    with open(path) as f:
        raw = json.load(f)
    acc_matrix = {int(k): v for k, v in raw["acc_matrix"].items()}
    memory_class_counts = {int(k): v for k, v in raw["memory_class_counts"].items()}
    return summarize(
        method, acc_matrix, memory_class_counts,
        gp_stats=raw.get("gradient_projection_stats"),
        shift_stats=raw.get("shift_detection_stats"),
    )


def run_er(subjects_data, memory_size, epochs_per_subject, new_batch_size,
            mem_batch_size, lr, device) -> dict:
    set_seed(SEED)
    model = MudviCNN()
    memory = ReservoirMemory(capacity=memory_size, seed=SEED)
    trainer = ContinualTrainer(
        model, memory, device, lr=lr, new_batch_size=new_batch_size,
        mem_batch_size=mem_batch_size, epochs_per_subject=epochs_per_subject,
        seed=SEED, use_gradient_projection=False, relationship_shift_detection=False,
    )
    acc_matrix = trainer.run(subjects_data)
    result = summarize("ER", acc_matrix, memory.class_counts())
    del trainer, model, memory
    gc.collect()
    return result


def run_mir(subjects_data, memory_size, epochs_per_subject, new_batch_size,
             mem_batch_size, lr, device) -> dict:
    set_seed(SEED)
    model = MudviCNN()
    memory = ReservoirMemory(capacity=memory_size, seed=SEED)
    trainer = MIRTrainer(
        model, memory, device, lr=lr, new_batch_size=new_batch_size,
        mem_batch_size=mem_batch_size, epochs_per_subject=epochs_per_subject,
        seed=SEED, use_gradient_projection=False, relationship_shift_detection=False,
    )
    acc_matrix = trainer.run(subjects_data)
    result = summarize("MIR", acc_matrix, memory.class_counts())
    del trainer, model, memory
    gc.collect()
    return result


def write_table(results: dict, out_dir: str):
    csv_path = os.path.join(out_dir, "comparison_table.csv")
    json_path = os.path.join(out_dir, "comparison_table.json")

    fieldnames = ["Method"] + [f"Acc({sid},{sid})" for sid in SUBJECTS] + \
        [f"Acc(N,{sid})" for sid in SUBJECTS] + \
        ["FinalAvgAcc", "BWT", "Forgetting"] + [f"MemClass{c}" for c in range(4)]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method in METHOD_ORDER:
            r = results[method]
            row = {"Method": method}
            for sid in SUBJECTS:
                row[f"Acc({sid},{sid})"] = r["acc_ii"][sid]
                row[f"Acc(N,{sid})"] = r["acc_Ni"][sid]
            row["FinalAvgAcc"] = r["final_avg_acc"]
            row["BWT"] = r["bwt"]
            row["Forgetting"] = r["forgetting"]
            for c in range(4):
                row[f"MemClass{c}"] = r["memory_class_counts"].get(c, r["memory_class_counts"].get(str(c), 0))
            writer.writerow(row)

    with open(json_path, "w") as f:
        json.dump({"subjects": SUBJECTS, "seed": SEED, "results": results}, f, indent=2)

    print(f"\nWrote unified table: {csv_path}")
    print(f"Wrote unified table: {json_path}")


def make_figures(results: dict, figures_dir: str):
    os.makedirs(figures_dir, exist_ok=True)

    # Figure 1: final average accuracy, all 6 methods.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vals = [results[m]["final_avg_acc"] for m in METHOD_ORDER]
    ax.bar(METHOD_ORDER, vals, color=[COLORS[m] for m in METHOD_ORDER])
    ax.set_ylabel("Final average accuracy")
    ax.set_title("Figure 1: Final Accuracy (A01→A02→A03)")
    ax.set_ylim(0, max(vals) * 1.25)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "fig1_final_accuracy.png"), dpi=200)
    plt.close(fig)

    # Figure 2: Acc(N,i) per subject, all 6 methods.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(SUBJECTS))
    width = 0.13
    for i, m in enumerate(METHOD_ORDER):
        vals = [results[m]["acc_Ni"][sid] for sid in SUBJECTS]
        ax.bar(x + (i - len(METHOD_ORDER) / 2) * width, vals, width,
               label=m, color=COLORS[m])
    ax.set_xticks(x)
    ax.set_xticklabels([f"A{sid}" for sid in SUBJECTS])
    ax.set_ylabel("Acc(N, i)")
    ax.set_title("Figure 2: Accuracy After Full Sequence, Per Subject")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "fig2_acc_N_i.png"), dpi=200)
    plt.close(fig)

    # Figure 3: BWT and Forgetting, all 6 methods, two panels.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bwt_vals = [results[m]["bwt"] for m in METHOD_ORDER]
    fgt_vals = [results[m]["forgetting"] for m in METHOD_ORDER]
    axes[0].bar(METHOD_ORDER, bwt_vals, color=[COLORS[m] for m in METHOD_ORDER])
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("BWT")
    axes[0].set_title("Backward Transfer")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(METHOD_ORDER, fgt_vals, color=[COLORS[m] for m in METHOD_ORDER])
    axes[1].set_ylabel("Forgetting")
    axes[1].set_title("Forgetting")
    axes[1].tick_params(axis="x", rotation=30)
    fig.suptitle("Figure 3: Continual-Learning Stability")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "fig3_bwt_forgetting.png"), dpi=200)
    plt.close(fig)

    # Figure 4: GP/RSD ablation, MUDVI variants only, accuracy + BWT.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    acc_vals = [results[m]["final_avg_acc"] for m in MUDVI_VARIANTS]
    bwt_vals = [results[m]["bwt"] for m in MUDVI_VARIANTS]
    axes[0].bar(MUDVI_VARIANTS, acc_vals, color=[COLORS[m] for m in MUDVI_VARIANTS])
    axes[0].set_ylabel("Final average accuracy")
    axes[0].set_title("Accuracy")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(MUDVI_VARIANTS, bwt_vals, color=[COLORS[m] for m in MUDVI_VARIANTS])
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("BWT")
    axes[1].set_title("Backward Transfer")
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle("Figure 4: GP / RSD Ablation (MUDVI variants only)")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "fig4_gp_rsd_ablation.png"), dpi=200)
    plt.close(fig)

    print(f"Wrote 4 figures to {figures_dir}")


def main():
    parser = argparse.ArgumentParser(description="6-way MUDVI/GP/RSD/ER/MIR comparison, A01-A03 only.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--stage1_dir", type=str, default=os.path.dirname(__file__),
                         help="Directory containing stage1_*.json (default: mudvi_baseline/).")
    parser.add_argument("--smoke", action="store_true",
                         help="Tiny epochs/memory for ER and MIR to sanity-check the pipeline end-to-end.")
    parser.add_argument("--rerun_mudvi", action="store_true",
                         help="Retrain the 4 MUDVI variants instead of reusing stage1_*.json (not used by default).")
    args = parser.parse_args()

    print(f"Running subjects: {SUBJECTS}")
    assert SUBJECTS == ["01", "02", "03"], "Refusing to run: SUBJECTS must be exactly A01-A03."

    memory_size = 20 if args.smoke else 200
    epochs_per_subject = 1 if args.smoke else 3
    new_batch_size = 32
    mem_batch_size = 32
    lr = 1e-3
    test_fraction = 0.2
    out_dir = os.path.join(args.out_dir, "smoke") if args.smoke else args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading and segmenting subjects: {SUBJECTS}")
    subjects_data = []
    for sid in SUBJECTS:
        sd = load_subject(sid, args.data_dir, test_fraction=test_fraction, seed=SEED)
        print(f"  A{sid}T: train segments={len(sd.y_train)}, test segments={len(sd.y_test)}")
        subjects_data.append(sd)

    results = {}

    if args.rerun_mudvi:
        for method in MUDVI_VARIANTS:
            set_seed(SEED)
            model = MudviCNN()
            memory = ClassBalancedMemory(capacity=memory_size, seed=SEED)
            trainer = ContinualTrainer(
                model, memory, device, lr=lr, new_batch_size=new_batch_size,
                mem_batch_size=mem_batch_size, epochs_per_subject=epochs_per_subject, seed=SEED,
                use_gradient_projection=("GP" in method),
                relationship_shift_detection=("RSD" in method),
            )
            print(f"\n=== Training {method} (fresh) ===")
            acc_matrix = trainer.run(subjects_data)
            gp_summary = trainer.gp_stats.summary() if trainer.gp_stats is not None else None
            shift_summary = None
            if trainer.relationship_shift_detection:
                log = trainer.shift_log
                shift_summary = {
                    "total_steps_logged": len(log),
                    "mmd_shifts": sum(1 for e in log if e["mmd_shift"]),
                    "confidence_shifts": sum(1 for e in log if e["confidence_shift"]),
                    "final_shifts": sum(1 for e in log if e["final_shift"]),
                    "confidence_only_shifts": sum(1 for e in log if e["confidence_shift"] and not e["mmd_shift"]),
                }
            results[method] = summarize(method, acc_matrix, memory.class_counts(), gp_summary, shift_summary)
            del trainer, model, memory
            gc.collect()
    else:
        print("\nReusing existing Stage-1 results for MUDVI / +GP / +RSD / +GP+RSD "
              "(protocol: A01-A03, 3 epochs/subject, memory=200, seed=0 -- "
              "from mudvi_baseline/stage1_*.json).")
        for method in MUDVI_VARIANTS:
            results[method] = load_stage1_result(args.stage1_dir, method)
            print(f"  loaded {method}: final_avg_acc={results[method]['final_avg_acc']:.4f}, "
                  f"bwt={results[method]['bwt']:.4f}")

    print("\n=== Training ER (fresh) ===")
    results["ER"] = run_er(subjects_data, memory_size, epochs_per_subject,
                            new_batch_size, mem_batch_size, lr, device)
    print(f"  ER: final_avg_acc={results['ER']['final_avg_acc']:.4f}, bwt={results['ER']['bwt']:.4f}")

    print("\n=== Training MIR (fresh) ===")
    results["MIR"] = run_mir(subjects_data, memory_size, epochs_per_subject,
                              new_batch_size, mem_batch_size, lr, device)
    print(f"  MIR: final_avg_acc={results['MIR']['final_avg_acc']:.4f}, bwt={results['MIR']['bwt']:.4f}")

    with open(os.path.join(out_dir, "er.json"), "w") as f:
        json.dump(results["ER"], f, indent=2)
    with open(os.path.join(out_dir, "mir.json"), "w") as f:
        json.dump(results["MIR"], f, indent=2)

    write_table(results, out_dir)
    make_figures(results, os.path.join(out_dir, "figures"))

    print("\n" + "=" * 70)
    print(f"DONE. Running subjects: {SUBJECTS} (confirmed, A04-A09 never touched).")
    print(f"{'Method':<16}{'FinalAvgAcc':<14}{'BWT':<10}{'Forgetting':<12}")
    for method in METHOD_ORDER:
        r = results[method]
        print(f"{method:<16}{r['final_avg_acc']:<14.4f}{r['bwt']:<10.4f}{r['forgetting']:<12.4f}")


if __name__ == "__main__":
    main()
