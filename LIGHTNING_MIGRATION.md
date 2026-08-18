# Lightning AI migration: what changed and how to run

This document covers the scaffolding added to move execution off the local
machine and onto Lightning AI. It does **not** port the training loop to
`pytorch_lightning.LightningModule`/`Trainer` — Lightning AI Studios/Jobs
run arbitrary Python, so the existing `torch`-native `ContinualTrainer`
(`mudvi_baseline/trainer.py`) is unchanged and still the source of truth
for every method. Rewriting it into the PyTorch Lightning framework's own
`Trainer` API would be a large, disruptive rewrite of already-validated,
paper-matched code for no requirement asked for — if you actually want
that rewrite (e.g. to get its built-in multi-GPU/logging conveniences),
say so explicitly and it can be scoped separately.

## 1. What's new

| File | Purpose |
|---|---|
| `requirements.txt`, `environment.yml` | Pinned dependencies, verified against the versions this codebase was built/tested against locally (Python 3.12.2, torch 2.12.1, mne 1.12.1, numpy 2.1.3, scipy 1.16.0, matplotlib 3.10.3). Torch's Linux pip wheels bundle CUDA, so the same pin gets GPU support automatically on a Lightning GPU instance. |
| `mudvi_baseline/config.py` | Single `ExperimentConfig` + CLI parser covering every method (`mudvi`/`er`/`mir`/`gmed`) and every hyperparameter, replacing hardcoded constants in `run_comparison.py` and the MUDVI-only flag set in `run_baseline.py`. |
| `mudvi_baseline/checkpoint.py` | Save/load/resume orchestration: one checkpoint per completed subject, atomic writes (`os.replace`), config-mismatch guard on resume. |
| `mudvi_baseline/run_experiment.py` | The new unified entry point. Use this for anything running on Lightning. `main(argv)` is a thin wrapper around `run(cfg: ExperimentConfig)`, so sweep drivers can call `run()` directly with a programmatically-built config. |
| `mudvi_baseline/gmed_baseline.py` | GMED (Gradient-based Memory Editing, Jin et al. 2020) baseline -- `GMEDTrainer`. |
| `mudvi_baseline/run_sensitivity.py` | Hyperparameter sensitivity sweep driver (Phase 3). |
| `mudvi_baseline/run_subject_order.py` | Subject-order robustness sweep driver (Phase 3). |
| `results/{baseline,mudvi_gp,mudvi_rsd,mudvi_gp_rsd,er,mir,gmed,sensitivity,subject_order}/` | Structured output tree (see `results/README.md`). |
| `state_dict()`/`load_state_dict()` added to: `ClassBalancedMemory`, `ReservoirMemory`, `ChangePointDetector`, `ConfidenceShiftDetector`, `OracleShiftDetector`, `ContinualTrainer` | The serialization layer checkpointing depends on. Purely additive — no existing method's behavior changed; `run_baseline.py` and `run_comparison.py` still work exactly as before. |

**Nothing existing was rewritten.** `run_baseline.py`, `run_comparison.py`,
`run_stage1.sh` (aside from de-hardcoding its path — see below), and every
training/model/memory/detector module's core logic are untouched.

## 2. Environment setup on Lightning AI

```bash
pip install -r requirements.txt
# or: conda env create -f environment.yml
```

Do not bump any of these versions without re-verifying `data.py`'s GDF
parsing (mne API surface) and the new checkpoint (de)serialization still
work — see the comment at the top of `requirements.txt`.

## 3. Dataset handling

The `A0kT.gdf` files must be uploaded to Lightning AI's persistent storage
(a Lightning AI Studio's `/teamspace/...` mount, or an attached dataset/
drive — use whatever your Lightning AI plan's persistent-storage mechanism
is; this project does not assume a specific one). No path in the codebase
is hardcoded to your local Windows path — `run_baseline.py`,
`run_comparison.py`, and `run_experiment.py` all take `--data_dir`
explicitly, and `run_experiment.py` additionally falls back to a
`BCICIV_DATA_DIR` environment variable:

```bash
export BCICIV_DATA_DIR=/teamspace/studios/this_studio/BCICIV_2a_gdf
```

`run_stage1.sh` previously hardcoded `c:/Users/boula/Downloads/BCICIV_2a_gdf`;
it now requires `BCICIV_DATA_DIR` to be set and fails loudly if it isn't,
rather than silently defaulting to a path that won't exist remotely.

## 4. Launching experiments

One CLI, every method, no source edits:

```bash
# Baseline MUDVI, full 9 subjects
python -m mudvi_baseline.run_experiment --method mudvi \
  --subjects 01,02,03,04,05,06,07,08,09 --memory_size 200 --seed 0

# MUDVI + Gradient Projection
python -m mudvi_baseline.run_experiment --method mudvi --gradient_projection \
  --subjects 01,02,03,04,05,06,07,08,09 --memory_size 200 --seed 0

# MUDVI + Relationship-Shift Detection
python -m mudvi_baseline.run_experiment --method mudvi --relationship_shift_detection \
  --subjects 01,02,03,04,05,06,07,08,09 --memory_size 200 --seed 0

# MUDVI + GP + RSD
python -m mudvi_baseline.run_experiment --method mudvi \
  --gradient_projection --relationship_shift_detection \
  --subjects 01,02,03,04,05,06,07,08,09 --memory_size 200 --seed 0

# ER / MIR / GMED (established baselines)
python -m mudvi_baseline.run_experiment --method er   --subjects 01,02,03,04,05,06,07,08,09 --memory_size 200 --seed 0
python -m mudvi_baseline.run_experiment --method mir  --subjects 01,02,03,04,05,06,07,08,09 --memory_size 200 --seed 0
python -m mudvi_baseline.run_experiment --method gmed --subjects 01,02,03,04,05,06,07,08,09 --memory_size 200 --seed 0
```

Full flag list: `python -m mudvi_baseline.run_experiment --help`.

### Hyperparameter sensitivity sweep

`run_sensitivity.py` takes every flag `run_experiment.py` does, plus
`--param`/`--values`, and runs one point per value with everything else
held fixed at the given base config:

```bash
python -m mudvi_baseline.run_sensitivity \
  --method mudvi --gradient_projection --relationship_shift_detection \
  --subjects 01,02,03,04,05,06,07,08,09 --memory_size 200 --seed 0 \
  --param memory_size --values 50,100,200,400
```

`--param` accepts any of: `memory_size`, `lr`, `epochs_per_subject`,
`new_batch_size`, `mem_batch_size`, `confidence_window_size`,
`confidence_min_segment_length`, `gmed_edit_lr`. Each point is a normal
`run_experiment.py`-style run (own `results/<subdir>/<run_id>/` folder,
resumable via its own checkpoints); a summary aggregating
bwt/forgetting/final_avg_acc across the sweep is written to
`results/sensitivity/<result_subdir>__<param>/summary.{json,csv}`. If the
sweep is interrupted and re-launched with the same command, points whose
`metrics.json` already exists are skipped rather than re-trained.

### Subject-order robustness sweep

`run_subject_order.py` similarly takes every `run_experiment.py` flag,
plus `--num_orders`/`--order_seed`, and runs the canonical `--subjects`
order plus `--num_orders` additional random permutations of the same
subject set, with the TRAINING seed (`--seed`) held fixed so order is the
only thing that varies:

```bash
python -m mudvi_baseline.run_subject_order \
  --method mudvi --gradient_projection --relationship_shift_detection \
  --subjects 01,02,03,04,05,06,07,08,09 --memory_size 200 --seed 0 \
  --num_orders 5 --order_seed 0
```

Summary (per-order bwt/forgetting/final_avg_acc plus mean/std across
orders) is written to `results/subject_order/<result_subdir>/summary.{json,csv}`.
Same restart behavior as `run_sensitivity.py`: already-finished orders are
skipped on re-launch.

Every run writes to `results/<result_subdir>/<run_id>/` (see
`results/README.md` for the folder mapping and `config.py`'s
`result_subdir()`/`run_id()` for the exact naming rule). `--run_name`
overrides the auto-generated run id if you want a specific folder name
(e.g. for the sensitivity/subject-order phases).

## 5. Checkpointing and restart

`--checkpoint_every N` (default 1) saves a checkpoint after every N
completed subjects; the granularity is per-subject, not per-batch — see
`checkpoint.py`'s module docstring for why (a subject-training phase takes
minutes, so losing at most one in-progress subject to a preemption is an
acceptable, well-scoped restart cost; per-batch checkpointing would add
I/O overhead for no practical benefit at this scale).

To resume a preempted run, re-issue the **identical** command with
`--resume` appended:

```bash
python -m mudvi_baseline.run_experiment --method mudvi --gradient_projection \
  --subjects 01,02,03,04,05,06,07,08,09 --memory_size 200 --seed 0 --resume
```

`run_experiment.py` refuses to resume if the checkpointed config doesn't
match the current flags (`checkpoint.assert_config_matches`) — this is
intentional: silently continuing a run under different hyperparameters
would invalidate the comparison without anyone noticing. Use a different
`--run_name` for a genuinely new run instead of forcing a mismatched
resume.

**Known limitation, disclosed rather than hidden:** the checkpoint restores
model weights, optimizer state, the full replay-memory buffer contents,
GP/RSD diagnostic state, and the trainer's own RNG stream exactly — but
not the memory buffer's *internal* reservoir-sampling RNG stream. A
resumed run is restartable and produces a valid continuation, but is not
guaranteed bit-identical to an uninterrupted run from that point on. See
`ContinualTrainer.state_dict()`'s docstring in `trainer.py`.

## 6. What this does not yet do

- **`run_comparison.py`'s 6-way A01–A03 comparison is not ported** to
  `run_experiment.py` — it still works standalone for its original purpose
  (a single combined table+figures artifact); `run_experiment.py` produces
  one run's metrics at a time, meant to be aggregated afterward across
  runs for the full comparison tables.
