"""Hyperparameter sensitivity sweep (Phase 3 of the experimental plan --
see `LIGHTNING_MIGRATION.md` and `results/README.md`, where
`results/sensitivity/` previously existed only as an empty destination
folder with no runner script).

One-factor-at-a-time sweep: holds every flag at a fixed base config and
varies a single hyperparameter across a list of values, reusing
`run_experiment.run()` for each point (not a CLI subprocess per point --
`dataclasses.replace` on one already-parsed `ExperimentConfig` avoids
round-tripping every variant through argv strings). Every flag
`run_experiment.py` accepts is available here too (same parser,
`config.build_arg_parser()`, plus `--param`/`--values`), so the base
config for the sweep is specified exactly like a normal run.

Usage (example: memory-size sensitivity for the full MUDVI+GP+RSD method):
  python -m mudvi_baseline.run_sensitivity \\
      --method mudvi --gradient_projection --relationship_shift_detection \\
      --subjects 01,02,03,04,05,06,07,08,09 --seed 0 \\
      --param memory_size --values 50,100,200,400

  # learning-rate sensitivity for ER:
  python -m mudvi_baseline.run_sensitivity --method er \\
      --subjects 01,02,03,04,05,06,07,08,09 --seed 0 \\
      --param lr --values 0.0001,0.0003,0.001,0.003,0.01

Each point lands in the normal `results/{result_subdir}/` layout (e.g.
`results/mudvi_gp_rsd/sens_memory_size_50_seed0_mem50_subj01-...-09/`),
`--run_name` prefixed with `sens_{param}_{value}` so it can't collide with
a plain `run_experiment.py` run of the same base config. After all points
finish, a summary aggregating bwt/forgetting/final_avg_acc across the
sweep is written to
`results/sensitivity/{result_subdir}__{param}/summary.{json,csv}`.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import replace

from .config import build_arg_parser, parse_config
from . import run_experiment

# Fields safe to sweep: numeric ExperimentConfig fields with a real effect
# on training. Restricted to an explicit allowlist (rather than any
# dataclass field) so a typo in --param fails clearly instead of silently
# overriding something like `seed` or `checkpoint_every`.
SWEEPABLE_PARAMS = {
    "memory_size": int,
    "lr": float,
    "epochs_per_subject": int,
    "new_batch_size": int,
    "mem_batch_size": int,
    "confidence_window_size": int,
    "confidence_min_segment_length": int,
    "gmed_edit_lr": float,
}


def build_parser():
    parser = build_arg_parser()
    parser.add_argument("--param", type=str, required=True, choices=sorted(SWEEPABLE_PARAMS),
                         help="ExperimentConfig field to sweep, holding every other flag fixed.")
    parser.add_argument("--values", type=str, required=True,
                         help="Comma-separated values for --param, e.g. 50,100,200,400.")
    return parser


def main(argv=None):
    parser = build_parser()
    args, _ = parser.parse_known_args(argv)
    param = args.param
    cast = SWEEPABLE_PARAMS[param]
    values = [cast(v) for v in args.values.split(",")]

    # Reuse config.parse_config for validation (data_dir presence, GP/RSD
    # vs. method compatibility) by re-parsing the same argv minus the two
    # sweep-only flags argparse doesn't know about on that path.
    base_argv = list(argv) if argv is not None else None
    base_cfg = parse_config(_strip_sweep_flags(base_argv))

    result_subdir = base_cfg.result_subdir()
    sweep_dir = os.path.join(base_cfg.out_dir, "sensitivity", f"{result_subdir}__{param}")
    os.makedirs(sweep_dir, exist_ok=True)

    rows = []
    for value in values:
        run_name = f"sens_{param}_{value}_seed{base_cfg.seed}_mem{base_cfg.memory_size}_subj{'-'.join(base_cfg.subjects)}"
        cfg = replace(base_cfg, **{param: value}, run_name=run_name)
        run_dir = os.path.join(cfg.out_dir, cfg.result_subdir(), cfg.run_id())
        metrics_path = os.path.join(run_dir, "metrics.json")
        if os.path.exists(metrics_path):
            # Sweep restarted after an interruption (e.g. a Lightning AI
            # Studio timeout): this point already finished a prior pass,
            # so reuse its metrics instead of re-training for hours.
            print(f"\n[run_sensitivity] === {param}={value}: already done, reusing {metrics_path} ===")
            with open(metrics_path) as f:
                metrics = json.load(f)
        else:
            print(f"\n[run_sensitivity] === {param}={value} -> {cfg.result_subdir()}/{cfg.run_id()} ===")
            metrics = run_experiment.run(cfg)
        rows.append({
            "param": param,
            "value": value,
            "bwt": metrics["bwt"],
            "forgetting": metrics["forgetting"],
            "final_avg_acc": metrics["final_avg_acc"],
            "elapsed_seconds": metrics["elapsed_seconds"],
            "run_dir": run_dir,
        })

    summary = {
        "method": base_cfg.method,
        "gradient_projection": base_cfg.gradient_projection,
        "relationship_shift_detection": base_cfg.relationship_shift_detection,
        "swept_param": param,
        "base_config": base_cfg.to_dict(),
        "points": rows,
    }
    with open(os.path.join(sweep_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(sweep_dir, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["param", "value", "bwt", "forgetting", "final_avg_acc",
                                                "elapsed_seconds", "run_dir"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[run_sensitivity] {len(rows)} points written to {sweep_dir}/summary.{{json,csv}}")
    for r in rows:
        print(f"  {param}={r['value']:<10} bwt={r['bwt']:.4f}  forgetting={r['forgetting']:.4f}  "
              f"final_avg_acc={r['final_avg_acc']:.4f}")
    return summary


def _strip_sweep_flags(argv):
    """Removes --param/--values (and their values) from argv before handing
    it to `config.parse_config`, which doesn't know those two flags."""
    if argv is None:
        import sys
        argv = sys.argv[1:]
    out = []
    skip_next = False
    for tok in argv:
        if skip_next:
            skip_next = False
            continue
        if tok in ("--param", "--values"):
            skip_next = True
            continue
        if tok.startswith("--param=") or tok.startswith("--values="):
            continue
        out.append(tok)
    return out


if __name__ == "__main__":
    main()
