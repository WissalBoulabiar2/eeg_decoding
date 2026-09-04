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
IMPLEMENTED_METHODS = {"mudvi", "er", "mir", "gmed", "joint"}
KNOWN_UNIMPLEMENTED_METHODS = set()
ALL_METHODS = IMPLEMENTED_METHODS | KNOWN_UNIMPLEMENTED_METHODS


@dataclass
class ExperimentConfig:
    method: str  # "mudvi" | "er" | "mir" | "gmed"
    data_dir: str
    subjects: list  # e.g. ["01", "02", "03"]
    dataset: str = "bci2a"  # "bci2a" | "high_gamma" -- see data_high_gamma.py
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

    # MUDVI-only extension: OCAR++ (Conflict-Aware Fisher Preconditioning,
    # ocarpp.py). This project's OWN extension of OCAR -- not part of the
    # published paper/repo. Reuses OCAR's K-FAC/Fisher machinery (same
    # ocar_alpha_ema/ocar_regul/ocar_fim_update_every hyperparameters
    # above) but adds a slow parameter-consolidation anchor and a
    # directional-conflict-aware protection factor; see ocarpp.py's
    # module docstring for the full mechanism. Mutually exclusive with
    # `ocar` (they are two separate methods: method="ocar" vs.
    # method="ocar++", selected by which flag is set) and with
    # `gradient_projection` (no such ablation is defined for OCAR++);
    # both are enforced in parse_config below and again in
    # ContinualTrainer.__init__.
    ocar_plusplus: bool = False
    ocarpp_beta_anchor: float = 0.999
    ocarpp_gamma: float = 1.0
    ocarpp_eps: float = 1e-8

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
        ocar, ocar_plusplus) to the results/ layout requested for the
        Lightning migration: results/{baseline,er,mir,gmed,mudvi_gp,
        mudvi_rsd,mudvi_gp_rsd,mudvi_ocar,mudvi_ocar_gp,mudvi_ocarpp}/.
        `ocar` combined with `relationship_shift_detection` (RSD stays
        diagnostic-only, see trainer.py, so this combination changes no
        accuracy result but is still given its own subfolder for
        run-isolation). `ocar` and `ocar_plusplus` are mutually
        exclusive (enforced in parse_config/ContinualTrainer)."""
        if self.method != "mudvi":
            return self.method  # results/er/, results/mir/, results/gmed/
        parts = ["mudvi"]
        if self.ocar:
            parts.append("ocar")
        if self.ocar_plusplus:
            parts.append("ocarpp")
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

    def run_dir(self) -> str:
        """results/[<dataset>/]<result_subdir>/<run_id>/ -- dataset is
        only inserted for non-default datasets, so BCI IV-2a's existing
        results/<result_subdir>/<run_id>/ layout (documented in
        LIGHTNING_MIGRATION.md) stays byte-identical. Shared by
        run_experiment.py and run_subject_order.py so both agree on where
        a given config's results live."""
        dataset_parts = [] if self.dataset == "bci2a" else [self.dataset]
        return os.path.join(self.out_dir, *dataset_parts, self.result_subdir(), self.run_id())

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
                         help="For --dataset bci2a: folder containing A0kT.gdf files. "
                              "For --dataset high_gamma: the HGD data/ folder containing "
                              "train/ and test/ subfolders. Falls back to the "
                              "BCICIV_DATA_DIR environment variable. Required (via flag "
                              "or env var) -- never hardcode this.")
    parser.add_argument("--dataset", type=str, default="bci2a", choices=["bci2a", "high_gamma"],
                         help="bci2a (default, Duan et al.'s BCI IV-2a) or high_gamma "
                              "(Schirrmeister et al.'s High Gamma Dataset -- not one of "
                              "Duan et al.'s three datasets, see data_high_gamma.py for "
                              "the implementation decisions this involves).")
    parser.add_argument("--subjects", type=str, default=",".join(DEFAULT_SUBJECT_ORDER),
                         help="Comma-separated subject order, e.g. 01,02,03 (bci2a) or "
                              "1,2,3 (high_gamma, unpadded 1..14). Default: bci2a's all 9.")
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

    parser.add_argument("--ocar_plusplus", action="store_true",
                         help="MUDVI only. OCAR++ (Conflict-Aware Fisher Preconditioning), this "
                              "project's own extension of OCAR -- see ocarpp.py. Mutually "
                              "exclusive with --ocar (separate methods) and with "
                              "--gradient_projection. Ignored for --method er/mir/gmed.")
    parser.add_argument("--ocarpp_beta_anchor", type=float, default=0.999,
                         help="EMA momentum beta_a for OCAR++'s slow consolidation anchor "
                              "theta*_t (higher = slower-moving anchor).")
    parser.add_argument("--ocarpp_gamma", type=float, default=1.0,
                         help="OCAR++ protection strength gamma in kappa_i = 1 + gamma * I_i * C_i.")
    parser.add_argument("--ocarpp_eps", type=float, default=1e-8,
                         help="OCAR++ numerical-stability epsilon used in both the directional "
                              "conflict c_i and the Fisher importance I_i.")

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
    if args.method in ("er", "mir", "gmed", "joint") and (
        args.gradient_projection or args.relationship_shift_detection or args.ocar
        or args.ocar_plusplus
    ):
        raise SystemExit(
            f"--gradient_projection/--relationship_shift_detection/--ocar/--ocar_plusplus are "
            f"MUDVI-only extensions and are not defined for --method {args.method}. "
            f"Remove them or switch to --method mudvi."
        )
    if args.ocar and args.ocar_plusplus:
        raise SystemExit(
            "--ocar and --ocar_plusplus are two separate methods (OCAR original vs. "
            "OCAR++) and cannot both be set. Choose one."
        )
    if args.ocar_plusplus and args.gradient_projection:
        raise SystemExit(
            "--ocar_plusplus does not support --gradient_projection (no such ablation "
            "is defined for OCAR++; OCAR original supports --ocar --gradient_projection "
            "instead)."
        )
    return ExperimentConfig(
        method=args.method,
        data_dir=args.data_dir,
        subjects=args.subjects.split(","),
        dataset=args.dataset,
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
        ocar_plusplus=args.ocar_plusplus,
        ocarpp_beta_anchor=args.ocarpp_beta_anchor,
        ocarpp_gamma=args.ocarpp_gamma,
        ocarpp_eps=args.ocarpp_eps,
        gmed_edit_lr=args.gmed_edit_lr,
        out_dir=args.out_dir,
        run_name=args.run_name,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )
