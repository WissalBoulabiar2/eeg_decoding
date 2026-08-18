"""Subject-order robustness sweep (Phase 3 of the experimental plan --
see `LIGHTNING_MIGRATION.md` and `results/README.md`, where
`results/subject_order/` previously existed only as an empty destination
folder with no runner script).

Runs the SAME method/config against several different subject arrival
orders -- the `--subjects` order given on the CLI (default: canonical
01..09) plus `--num_orders` additional random permutations of that same
subject set -- to check whether BWT/forgetting/final accuracy are an
artifact of subject arrival order rather than the method itself. The
TRAINING seed (`--seed`, controls weight init / batch shuffling / memory
sampling) is held fixed across every order so order is the only thing
that varies between runs; a separate `--order_seed` controls only which
permutations are generated.

Usage:
  python -m mudvi_baseline.run_subject_order \\
      --method mudvi --gradient_projection --relationship_shift_detection \\
      --subjects 01,02,03,04,05,06,07,08,09 --seed 0 \\
      --num_orders 5 --order_seed 0

Each point lands in the normal `results/{result_subdir}/` layout (e.g.
`results/mudvi_gp_rsd/subjorder_02_seed0_mem200_subj04-...`), run-named
`subjorder_{index}_...` so it can't collide with a plain run of the same
base config. After all points finish, a summary aggregating
bwt/forgetting/final_avg_acc (mean/std across orders) is written to
`results/subject_order/{result_subdir}/summary.{json,csv}`.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import replace

import numpy as np

from .config import build_arg_parser, parse_config
from . import run_experiment


def build_parser():
    parser = build_arg_parser()
    parser.add_argument("--num_orders", type=int, default=5,
                         help="Number of additional random subject-order permutations to run, "
                              "on top of the canonical order given via --subjects.")
    parser.add_argument("--order_seed", type=int, default=0,
                         help="Seed for generating the random permutations (independent of --seed, "
                              "which controls training randomness and is held fixed across orders).")
    return parser


def _distinct_permutations(canonical: list, num_orders: int, order_seed: int) -> list:
    """Canonical order first, then up to `num_orders` distinct random
    permutations of the same subject set (retried on collision -- with 9
    subjects there are 9! = 362880 orderings, so collisions are rare, but
    guarded rather than assumed away)."""
    rng = np.random.RandomState(order_seed)
    seen = {tuple(canonical)}
    orders = [list(canonical)]
    attempts = 0
    while len(orders) < 1 + num_orders and attempts < 100 * num_orders + 100:
        attempts += 1
        perm = [str(s) for s in rng.permutation(canonical)]
        key = tuple(perm)
        if key in seen:
            continue
        seen.add(key)
        orders.append(perm)
    return orders


def main(argv=None):
    parser = build_parser()
    args, _ = parser.parse_known_args(argv)
    num_orders, order_seed = args.num_orders, args.order_seed

    base_argv = list(argv) if argv is not None else None
    base_cfg = parse_config(_strip_sweep_flags(base_argv))

    orders = _distinct_permutations(base_cfg.subjects, num_orders, order_seed)

    result_subdir = base_cfg.result_subdir()
    sweep_dir = os.path.join(base_cfg.out_dir, "subject_order", result_subdir)
    os.makedirs(sweep_dir, exist_ok=True)

    rows = []
    for i, order in enumerate(orders):
        run_name = f"subjorder_{i:02d}_seed{base_cfg.seed}_mem{base_cfg.memory_size}_subj{'-'.join(order)}"
        cfg = replace(base_cfg, subjects=order, run_name=run_name)
        run_dir = os.path.join(cfg.out_dir, cfg.result_subdir(), cfg.run_id())
        metrics_path = os.path.join(run_dir, "metrics.json")
        if os.path.exists(metrics_path):
            # Sweep restarted after an interruption (e.g. a Lightning AI
            # Studio timeout): this order already finished a prior pass,
            # so reuse its metrics instead of re-training for hours.
            print(f"\n[run_subject_order] === order {i}: already done, reusing {metrics_path} ===")
            with open(metrics_path) as f:
                metrics = json.load(f)
        else:
            print(f"\n[run_subject_order] === order {i} ({'canonical' if i == 0 else 'random'}): "
                  f"{'-'.join(order)} -> {cfg.result_subdir()}/{cfg.run_id()} ===")
            metrics = run_experiment.run(cfg)
        rows.append({
            "order_index": i,
            "is_canonical": i == 0,
            "subject_order": "-".join(order),
            "bwt": metrics["bwt"],
            "forgetting": metrics["forgetting"],
            "final_avg_acc": metrics["final_avg_acc"],
            "elapsed_seconds": metrics["elapsed_seconds"],
            "run_dir": run_dir,
        })

    bwts = [r["bwt"] for r in rows]
    forgettings = [r["forgetting"] for r in rows]
    accs = [r["final_avg_acc"] for r in rows]
    summary = {
        "method": base_cfg.method,
        "gradient_projection": base_cfg.gradient_projection,
        "relationship_shift_detection": base_cfg.relationship_shift_detection,
        "training_seed": base_cfg.seed,
        "order_seed": order_seed,
        "num_orders": len(rows),
        "base_config": base_cfg.to_dict(),
        "points": rows,
        "aggregate": {
            "bwt_mean": float(np.mean(bwts)), "bwt_std": float(np.std(bwts)),
            "forgetting_mean": float(np.mean(forgettings)), "forgetting_std": float(np.std(forgettings)),
            "final_avg_acc_mean": float(np.mean(accs)), "final_avg_acc_std": float(np.std(accs)),
        },
    }
    with open(os.path.join(sweep_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(sweep_dir, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["order_index", "is_canonical", "subject_order", "bwt",
                                                "forgetting", "final_avg_acc", "elapsed_seconds", "run_dir"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[run_subject_order] {len(rows)} orders written to {sweep_dir}/summary.{{json,csv}}")
    print(f"  BWT:            {summary['aggregate']['bwt_mean']:.4f} +/- {summary['aggregate']['bwt_std']:.4f}")
    print(f"  Forgetting:     {summary['aggregate']['forgetting_mean']:.4f} +/- {summary['aggregate']['forgetting_std']:.4f}")
    print(f"  FinalAvgAcc:    {summary['aggregate']['final_avg_acc_mean']:.4f} +/- {summary['aggregate']['final_avg_acc_std']:.4f}")
    return summary


def _strip_sweep_flags(argv):
    """Removes --num_orders/--order_seed (and their values) from argv
    before handing it to `config.parse_config`, which doesn't know those
    two flags."""
    if argv is None:
        import sys
        argv = sys.argv[1:]
    out = []
    skip_next = False
    for tok in argv:
        if skip_next:
            skip_next = False
            continue
        if tok in ("--num_orders", "--order_seed"):
            skip_next = True
            continue
        if tok.startswith("--num_orders=") or tok.startswith("--order_seed="):
            continue
        out.append(tok)
    return out


if __name__ == "__main__":
    main()
