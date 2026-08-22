"""Unified experiment configuration for Lightning AI (remote) execution.

Consolidates the flags previously split across `run_baseline.py` (MUDVI /
+GP / +RSD / +GP+RSD only) and `run_comparison.py` (hardcoded A01-A03,
ER/MIR only) into one CLI surface, per the Lightning AI migration
requirement to launch any method without editing source:

  python -m mudvi_baseline.run_experiment --method mudvi --gradient_projection \\
      --subjects 01,02,03 --memory_size 200 --seed 42

This module owns only argument parsing and the resulting config object; it
does not import torch or construct any model/trainer, so `--help` stays
fast and this file has no GPU/CPU-backend dependency.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, asdict

DEFAULT_SUBJECT_ORDER = [f"{i:02d}" for i in range(1, 10)]  # 01..09

# Methods with a working implementation. Kept as an explicit, checkable
# set rather than a silent KeyError, so an unknown method fails with a
# clear message instead of a confusing traceback deep in
# run_experiment.py. GMED (gmed_baseline.GMEDTrainer) was the last method
# added -- see IMPLEMENTATION_MAPPING.md for what each method covers.
IMPLEMENTED_METHODS = {"mudvi", "er", "mir", "gmed"}
KNOWN_UNIMPLEMENTED_METHODS = set()
ALL_METHODS = IMPLEMENTED_METHODS | KNOWN_UNIMPLEMENTED_METHODS


@dataclass
class ExperimentConfig:
    method: str  # "mudvi" | "er" | "mir" | "gmed"
    data_dir: str
    subjects: list  # e.g. ["01", "02", "03"]
    memory_size: int = 200
    epochs_per_subject: int = 15
    new_batch_size: int = 32
    mem_batch_size: int = 32
    lr: float = 1e-3
    test_fraction: float = 0.2
    seed: int = 0

    # MUDVI-only extensions (Additions 1 & 2). Ignored (and asserted off)
    # for method in {"er", "mir", "gmed"} -- these are MUDVI-specific,
    # not properties of the other baselines, per the fair-comparison
    # protocol (Section 9 of the experimental brief): ER/MIR/GMED are
    # compared against MUDVI variants, not hybridized with them.
    gradient_projection: bool = False
    relationship_shift_detection: bool = False
    confidence_signal_type: str = "loss"
    confidence_window_size: int = 40
    confidence_min_segment_length: int = 5

    # MUDVI-only extension: OCAR (Online Curvature-Aware Replay, Urettini
    # & Carta, ICML 2025 -- see ocar.py, verified against the official
    # repo github.com/edo-urettini/CL_stability). Same "MUDVI-specific,
    # not hybridized into er/mir/gmed" convention as gradient_projection/
    # relationship_shift_detection above. Can be combined with
    # gradient_projection (the "MUDVI+OCAR+GP" ablation condition, an
    # extension beyond the published algorithm -- see
    # trainer.ContinualTrainer._train_step_ocar). Defaults match the
    # official repo's published "robust_grad" config exactly
    # (alpha_ema=1.0, regul=0.01); ocar_fim_update_every=1 is this
    # project's disclosed adaptation of their train_epochs-based gating
    # (see ocar.py's OCARPreconditioner.maybe_update_fisher docstring).
    ocar: bool = False
    ocar_alpha_ema: float = 1.0
    ocar_regul: float = 0.01
    ocar_fim_update_every: int = 1

    # GMED-only extension (gmed_baseline.GMEDTrainer). Ignored for every
    # other method, same convention as the MUDVI-only fields above.
    gmed_edit_lr: float = 0.1

    # Lightning AI / remote-execution plumbing.
    out_dir: str = "results"
    run_name: str | None = None  # default derived from method+flags, see run_experiment.py
    checkpoint_every: int = 1  # checkpoint after every N completed subjects; 0 disables
    resume: bool = False

    def result_subdir(self) -> str:
        """Maps (method, gradient_projection, relationship_shift_detection,
        ocar) to the results/ layout requested for the Lightning
        migration: results/{baseline,er,mir,gmed,mudvi_gp,mudvi_rsd,
        mudvi_gp_rsd,mudvi_ocar,mudvi_ocar_gp}/. `ocar` combined with
        `relationship_shift_detection` (RSD stays diagnostic-only, see
        trainer.py, so this combination changes no accuracy result but is
        still given its own subfolder for run-isolation)."""
        if self.method != "mudvi":
            return self.method  # results/er/, results/mir/, results/gmed/
        parts = ["mudvi"]
        if self.ocar:
            parts.append("ocar")
        if self.gradient_projection:
            parts.append("gp")
        if self.relationship_shift_detection:
            parts.append("rsd")
        if len(parts) == 1:
            return "baseline"
        return "_".join(parts)

    def run_id(self) -> str:
        if self.run_name:
            return self.run_name
        subj = "-".join(self.subjects)
        return f"{self.result_subdir()}_seed{self.seed}_mem{self.memory_size}_subj{subj}"

    def to_dict(self) -> dict:
        return asdict(self)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified continual-EEG experiment launcher (BCI IV 2a). "
                     "Selects method and hyperparameters entirely via flags -- "
                     "no source edits needed, per the Lightning AI migration plan."
    )
    parser.add_argument("--method", type=str, required=True, choices=sorted(ALL_METHODS),
                         help="mudvi (with optional --gradient_projection / "
                              "--relationship_shift_detection), er, mir, or gmed.")
    parser.add_argument("--data_dir", type=str,
                         default=os.environ.get("BCICIV_DATA_DIR"),
                         help="Path to the folder containing A0kT.gdf files. "
                              "Falls back to the BCICIV_DATA_DIR environment variable. "
                              "Required (via flag or env var) -- never hardcode this.")
    parser.add_argument("--subjects", type=str, default=",".join(DEFAULT_SUBJECT_ORDER),
                         help="Comma-separated subject order, e.g. 01,02,03. Default: all 9, in order.")
    parser.add_argument("--memory_size", type=int, default=200)
    parser.add_argument("--epochs_per_subject", type=int, default=15)
    parser.add_argument("--new_batch_size", type=int, default=32)
    parser.add_argument("--mem_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--gradient_projection", action="store_true",
                         help="MUDVI only (Addition 1). Ignored for --method er/mir/gmed.")
    parser.add_argument("--relationship_shift_detection", action="store_true",
                         help="MUDVI only (Addition 2). Ignored for --method er/mir/gmed.")
    parser.add_argument("--confidence_signal_type", type=str, default="loss",
                         choices=["loss", "confidence"])
    parser.add_argument("--confidence_window_size", type=int, default=40)
    parser.add_argument("--confidence_min_segment_length", type=int, default=5)

    parser.add_argument("--ocar", action="store_true",
                         help="MUDVI only. Online Curvature-Aware Replay (Urettini & Carta, "
                              "ICML 2025) -- K-FAC curvature-preconditioned replay update, see "
                              "ocar.py. Ignored for --method er/mir/gmed. May be combined with "
                              "--gradient_projection (the MUDVI+OCAR+GP ablation condition).")
    parser.add_argument("--ocar_alpha_ema", type=float, default=1.0,
                         help="Weight on the NEWLY computed Fisher estimate when blending with "
                              "the previous one (1.0 = no blending, matches the official repo's "
                              "published default; lower values enable EMA blending for ablation).")
    parser.add_argument("--ocar_regul", type=float, default=0.01,
                         help="Amount the K-FAC damping term tau grows by every Fisher-recompute "
                              "cycle (matches the official repo's published default).")
    parser.add_argument("--ocar_fim_update_every", type=int, default=1,
                         help="Recompute the K-FAC curvature estimate every N training steps "
                              "(default 1 = every step; see ocar.py's docstring for why the "
                              "official repo's literal train_epochs-based gating does not "
                              "transfer directly to this project's epochs_per_subject).")

    parser.add_argument("--gmed_edit_lr", type=float, default=0.1,
                         help="GMED only (gradient-ascent input-space edit step size). "
                              "Ignored for --method mudvi/er/mir.")

    parser.add_argument("--out_dir", type=str, default="results",
                         help="Root results directory; method/flags determine the subfolder "
                              "(results/baseline, results/mudvi_gp, results/er, ...).")
    parser.add_argument("--run_name", type=str, default=None,
                         help="Overrides the auto-generated run id used for the results subfolder name.")
    parser.add_argument("--checkpoint_every", type=int, default=1,
                         help="Save a checkpoint after every N completed subjects (0 disables checkpointing).")
    parser.add_argument("--resume", action="store_true",
                         help="Resume from the latest checkpoint in this run's results subfolder, if one exists.")
    return parser


def parse_config(argv=None) -> ExperimentConfig:
    args = build_arg_parser().parse_args(argv)
    if not args.data_dir:
        raise SystemExit(
            "No dataset path given: pass --data_dir or set the BCICIV_DATA_DIR "
            "environment variable (see LIGHTNING_MIGRATION.md)."
        )
    if args.method in ("er", "mir", "gmed") and (
        args.gradient_projection or args.relationship_shift_detection or args.ocar
    ):
        raise SystemExit(
            f"--gradient_projection/--relationship_shift_detection/--ocar are MUDVI-only "
            f"extensions and are not defined for --method {args.method}. "
            f"Remove them or switch to --method mudvi."
        )
    return ExperimentConfig(
        method=args.method,
        data_dir=args.data_dir,
        subjects=args.subjects.split(","),
        memory_size=args.memory_size,
        epochs_per_subject=args.epochs_per_subject,
        new_batch_size=args.new_batch_size,
        mem_batch_size=args.mem_batch_size,
        lr=args.lr,
        test_fraction=args.test_fraction,
        seed=args.seed,
        gradient_projection=args.gradient_projection,
        relationship_shift_detection=args.relationship_shift_detection,
        confidence_signal_type=args.confidence_signal_type,
        confidence_window_size=args.confidence_window_size,
        confidence_min_segment_length=args.confidence_min_segment_length,
        ocar=args.ocar,
        ocar_alpha_ema=args.ocar_alpha_ema,
        ocar_regul=args.ocar_regul,
        ocar_fim_update_every=args.ocar_fim_update_every,
        gmed_edit_lr=args.gmed_edit_lr,
        out_dir=args.out_dir,
        run_name=args.run_name,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )
