# Results layout

Written by `mudvi_baseline/run_experiment.py` (see `LIGHTNING_MIGRATION.md`
at the repo root). Each method/configuration gets its own subfolder,
matching `config.ExperimentConfig.result_subdir()`:

| Folder | Populated by |
|---|---|
| `baseline/` | `--method mudvi` (no `--gradient_projection`, no `--relationship_shift_detection`) |
| `mudvi_gp/` | `--method mudvi --gradient_projection` |
| `mudvi_rsd/` | `--method mudvi --relationship_shift_detection` |
| `mudvi_gp_rsd/` | `--method mudvi --gradient_projection --relationship_shift_detection` |
| `er/` | `--method er` |
| `mir/` | `--method mir` |
| `gmed/` | `--method gmed` |
| `sensitivity/` | `mudvi_baseline/run_sensitivity.py` -- one `<result_subdir>__<param>/summary.{json,csv}` per sweep |
| `subject_order/` | `mudvi_baseline/run_subject_order.py` -- one `<result_subdir>/summary.{json,csv}` per sweep |

Note: `sensitivity/` and `subject_order/` hold only the sweep drivers'
aggregated `summary.{json,csv}`. Each individual sweep POINT is still a
normal run and lands in its own method folder above (e.g. a
`run_sensitivity.py --method mudvi --gradient_projection
--relationship_shift_detection --param memory_size` sweep's points live
under `mudvi_gp_rsd/sens_memory_size_*_.../`, not under `sensitivity/`
itself) -- so every run, swept or not, is still resumable/checkpointed
the same way.

Within each method folder, one subfolder per run (`config.ExperimentConfig.run_id()`,
e.g. `mudvi_gp_seed0_mem200_subj01-02-03-04-05-06-07-08-09/`), containing:

- `config.json` -- the fully resolved experiment configuration
- `metrics.json` -- acc_matrix, BWT, forgetting, memory class counts, GP/RSD
  diagnostic summaries (whichever apply to that run)
- `checkpoints/step_NN.pt` -- one checkpoint per completed subject, for resume

## Pre-existing files in this directory

`comparison_table.csv`, `comparison_table.json`, `er.json`, `mir.json`,
`figures/`, and `smoke/` were produced by the earlier `run_comparison.py`
script (A01-A03 only, 6-way MUDVI/GP/RSD/GP+RSD/ER/MIR comparison for
Prof. Duan's feedback). They are left in place, not moved, and
`run_comparison.py` still works standalone; `run_experiment.py` is the
newer, Lightning-AI-facing entry point and does not read or write these
particular files.
