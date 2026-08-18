# IMPLEMENTATION_MAPPING.md

Traceability from source papers to code, produced BEFORE implementation as required.
Hierarchy of authority: (1) Duan et al. MUDVI paper `duanWLDAYT24nn.pdf`, (2) `mudvi_gp_report.pdf`
(covers BOTH proposed additions — it is one self-contained report, not two separate papers), (3) existing
repo (none existed prior to this work), (4) general implementation knowledge (used only where explicitly
flagged as "IMPLEMENTATION DECISION" below).

---

## 0. Dataset facts confirmed by direct GDF inspection (mne 1.12.1, `read_raw_gdf`)

- 25 recorded channels: 22 EEG (`EEG-Fz, EEG-0..EEG-16, EEG-C3, EEG-Cz, EEG-C4, EEG-Pz`) + 3 EOG
  (`EOG-left, EOG-central, EOG-right`). Only the 22 EEG channels are used, matching "22 EEG channels" in
  the paper/task. EOG channels are dropped, not used for artifact regression (not mentioned in Duan §4.3).
- Sampling rate: 250 Hz, matches paper.
- T-file event codes: `768`=trial start/cue onset, `769`=Left hand, `770`=Right hand, `771`=Both feet,
  `772`=Tongue, `1023`=rejected trial (artifact marker), `32766`=new run. 288 trials/session (6 runs × 48).
- E-file event codes: cue class is `783` ("unknown") — true labels are NOT encoded in the GDF and the
  labeled `.mat` files are not present in this dataset folder. Per user decision, **E files are excluded
  from this implementation entirely**; evaluation uses a held-out split of each `A0kT` session instead.

---

## 1. Duan et al. (MUDVI) — Baseline

| Paper location | Operation | Code location |
|---|---|---|
| Def. 1, §3 (Continual EEG decoding) | Streaming subjects S1..SJ, sequential arrival, no revisiting except via memory | `src/mudvi/trainer.py` (main loop over subjects A01T→A09T) |
| §4.3 "Data Processing" (BCI IV-2a) | Segment size 400×22, stride 50, window t=3s–6s post-cue (750 samples @250Hz) → 8 segments/trial | `src/data/preprocessing.py: segment_trial()` |
| §4.3 "Model Settings" | 3-layer CNN: (1) temporal conv filter (1,C); (2) depthwise conv, temporal-specific spatial filters, filter (2,32); (3) pointwise conv; zero-padding between layers | `src/model/eeg_model.py: MudviCNN` |
| Eq. 1–2, Alg. 1 (Memory Update on Data Volume and Informativeness) | Cluster-level replace probability `A_C`, `P_t^C`; per-cluster hierarchical buffer, one cluster per detected subject | `src/mudvi/memory.py: MudviMemory.update()` |
| Eq. 3 (`P_t^in`) | Probability of moving incoming datum into memory, using `J_t=(1-n_{L_t}/n)I_t` and average memory importance `I_M`, hyperparameter λ=1 | `src/mudvi/memory.py: MudviMemory.p_in()` |
| §3.2 Eq. 4 (`q_t^C`), `I_C` def. | Cluster importance = mean gradient-norm of R=6 random representatives; sampling probability for joint-training batch construction | `src/mudvi/memory.py: MudviMemory.sample_batch()` |
| §3.3 Eq. 8–11 (kernel MMD subject-shift detector) | Moving-average feature `e_t=αf_t+(1−α)e_{t−1}`; distance metric `d_t` of dim m=8 vs. previous m steps; MMD² via U-statistic (Eq. 9–10) between two adjacent windows of size B; `L_t` increments when `δ_t>h` | `src/mudvi/mmd_detector.py: MMDSubjectShiftDetector` |
| §3.3 Eq. 12 (adaptive threshold) | `μ_t,μ_t^(2),σ_t` EMA with ρ=0.2; `h=μ_t+aσ_t`, a=1.96 (95% CI) | `src/mudvi/mmd_detector.py: AdaptiveThreshold` |
| Alg. 1 (full memory-update loop) | Orchestrates shift-detected cluster creation → free-space check → `P_t^in` gate → replacement | `src/mudvi/trainer.py: MudviTrainer.step()` |
| §4.2 (metrics) | Accuracy after sequential learning ends, BWT = mean(a_{N,i} − a_{i,i}) | `src/evaluation/metrics.py` |

**IMPLEMENTATION DECISIONS (not specified, or specified only for other datasets, in Duan et al.):**
- Train/test split per subject for computing Acc(i,i)/Acc(N,i): paper's Appendix E train/test protocol
  (A0XT train / A0XE test) is only described for the *joint-training upper bound*. Since E-file labels are
  unavailable here (§0), we hold out a stratified 20% of each `A0kT` session's **trials** (not segments, to
  avoid leakage across the 8 overlapping segments of one trial) as that subject's permanent test set. This
  mirrors the ~1:1 / 20%-test spirit used elsewhere in the paper (DEAP: 20% test) without inventing new
  ratios.
- Artifact rejection using GDF code `1023`: not mentioned anywhere in Duan et al. Implemented as an
  **opt-in, default-OFF** flag (`preprocessing.drop_artifact_trials`) so the baseline stays maximally
  faithful by default; logged explicitly when enabled.
- Bandpass filtering: not mentioned in Duan et al. §4.3 for BCI IV-2a. None applied by default (raw µV
  signal, per-segment z-normalized only, since some normalization is implicit in using gradient norms as
  importance scores and is standard/required before any CNN training). Flagged as an implementation
  decision in `preprocessing.py` docstring.
- `MudviCNN` exact layer widths/kernel counts beyond what §4.3 specifies (they give filter *shapes*, not
  channel counts) — chosen to match EEGNet-style channel counts referenced as the architectural family
  (Duan et al. cite EEGNet [10] as the CNN literature this design follows). Documented in
  `eeg_model.py` header.
- NOT/GLR interval-sampling scheme, number of random intervals, and detection threshold ζ for the
  change-point detector are not numerically specified in `mudvi_gp_report.pdf` beyond the GLR statistic
  itself (Eq. 8) — see §3 below.

---

## 2. `mudvi_gp_report.pdf` — Addition 1: Constrained Replay via Gradient Projection

| Paper location | Operation | Code location |
|---|---|---|
| §3.2 Eq. 5 | Compute `g*_k = ∇_θ R(θ,M)`, `g_{k+1} = ∇_θ R(θ,D_{k+1})` separately, both cross-entropy, from the SAME `θ_hat_k` | `src/proposed/gradient_projection.py: compute_separate_grads()` |
| §3.2, Eq. 3 (magnitude-aware projection) | If `⟨g*_k,g_{k+1}⟩<0`: larger-magnitude gradient has its component along the smaller-magnitude gradient removed (orthogonal projection); output is that single modified gradient (NOT a sum). If `⟨·,·⟩≥0`: `g̃=g*_k+g_{k+1}` | `src/proposed/gradient_projection.py: project_gradient()` |
| §4 Algorithm 1, lines 13–21 | One `optimizer.step()` per iteration using `g̃`; assigned via `.grad` then single step | `src/mudvi/trainer.py: MudviTrainer._train_step()` (branches on `use_gradient_projection`) |
| §4 "log conflicts" (task §8) | conflict count/%, dot product before/after projection, memory loss before/after | `src/proposed/gradient_projection.py: GradientProjectionStats` + `src/evaluation/metrics.py` |

**Verified invariant** (derived directly from Eq. 3, not invented): both branches of the projection
guarantee `⟨g̃,g*_k⟩≥0` — branch 1 gives exactly 0 by construction of orthogonal projection; branch 2 gives
`‖g*_k‖²−⟨g*_k,g_{k+1}⟩²/‖g_{k+1}‖²≥0` by Cauchy–Schwarz. Unit tests assert this numerically.

---

## 3. `mudvi_gp_report.pdf` — Addition 2: Confidence-Based Relationship-Shift Detection

| Paper location | Operation | Code location |
|---|---|---|
| §3.3, text | Freeze `θ_hat_{D1}` = deep copy of model after training on Task 1 (A01T), never updated again | `src/proposed/confidence_detector.py: freeze_reference_model()` |
| §3.3 Eq. 6 | `c_t = ℓ_t(θ_hat_{D1}) = -log[f(x_t;θ_hat_{D1})]_{y_t}` — labeled stream, `signal_type="loss"` (**main experiment**, labels available offline) | `src/proposed/confidence_detector.py: ConfidenceStream.compute(signal_type="loss")` |
| §3.3 Eq. 7 | `c_t = max_k [f(x_t;θ_hat_{D1})]_k` — label-free alternative, `signal_type="confidence"` | same file, `signal_type="confidence"` branch |
| §3.3 Eq. 8 (NOT/GLR statistic) | `R^c_{(s,e)}(c)=2log[sup_{η1,η2}L(...)L(...) / sup_η L(...)]` applied to the scalar stream `c_1,c_2,...`, Gaussian likelihood, one-line vs. two-line fit | `src/mudvi/change_point.py: not_glr_statistic()`, `ChangePointDetector.detect()` |
| §4 Algorithm 1, line 1 | `final_shift = MMD_shift OR Confidence_shift`; both detectors run independently, side by side, neither replaces the other | `src/mudvi/trainer.py: MudviTrainer._detect_shift()` |

**IMPLEMENTATION DECISIONS (GLR statistic given, surrounding procedure not numerically specified):**
- NOT (Narrowest-Over-Threshold) is applied here as: maintain a sliding window of the last `W` values of
  `c_t` (`W` configurable, default 40); at each step compute Eq. 8 for every interior split point `c` in
  the window and take the max; flag a change-point if `max_c R^c > ζ`, with `ζ` set via a chi-square
  critical value (asymptotic null distribution of a Gaussian GLR statistic, 2 extra free parameters →
  `ζ=χ²_{0.99,df=2}`) rather than an arbitrary constant. This is standard NOT/GLR practice
  (Baranowski, Chen & Fryzlewicz 2019) applied because the report gives the statistic but not these
  operational parameters; documented in the module docstring, not attributed to either paper as a given.
- Per §14 of the task and §3.4 of the report ("What Remains Unchanged"), a detected shift from **either**
  detector never automatically modifies the memory buffer in Experiments 1–4 — detection is purely
  diagnostic/logged, exactly as specified.

---

## 4. Experiment configuration matrix (task §15–16)

| Experiment | `use_gradient_projection` | `use_confidence_detector` | Memory | MMD detector |
|---|---|---|---|---|
| 1 — Baseline MUDVI | False | False | original | original |
| 2 — MUDVI + GP | True | False | original (unchanged) | original (unchanged) |
| 3 — MUDVI + Confidence | False | True | original (unchanged) | original (unchanged) |
| 4 — MUDVI + GP + Confidence | True | True | original (unchanged) | original (unchanged) |

All four share one training loop (`src/mudvi/trainer.py`) gated by these two booleans — no duplicated
training code, per task §20.

---

## 5. Established-baseline comparison (Prof. Duan feedback, requested ER/MIR/GMED) — standard literature, not from either source paper

Neither ER, MIR, nor GMED appears in Duan et al. or `mudvi_gp_report.pdf`; all three are standard
formulations from the continual-learning literature, requested directly by Prof. Duan's review feedback
("more detailed comparison with the established baselines such as ER and MIR").

| Source | Operation | Code location |
|---|---|---|
| Rolnick et al. 2019, "Experience Replay for Continual Learning"; Chaudhry et al. 2019, "On Tiny Episodic Memories" | Single global reservoir buffer (Vitter 1985), uniform replay sampling, interleaved with the incoming stream via one combined-loss update per step | `mudvi_baseline/er_baseline.py: ReservoirMemory` (reuses `trainer.ContinualTrainer`'s existing `_train_step_baseline` unmodified — that update rule already *is* standard ER) |
| Aljundi et al. 2019, "Online Continual Learning with Maximally Interfered Retrieval" (NeurIPS) | One-step virtual update using the incoming mini-batch's gradient; replay the buffer samples with the largest loss increase under that virtual update | `mudvi_baseline/mir_baseline.py: MIRTrainer._select_interfered()` |
| Jin et al. 2020, "Gradient Based Memory Editing for Task-Free Continual Learning" (arXiv:2006.15294) | Same one-step virtual update as MIR; instead of only selecting which buffered examples to replay, edits their INPUT in place via one gradient-ascent step under the virtual model, persisting the edit into the buffer only if it increased that example's loss | `mudvi_baseline/gmed_baseline.py: GMEDTrainer._edit_memory_batch()`, `train_on_subject()` |

**IMPLEMENTATION DECISIONS (standard efficient approximations, not tuned to favor either baseline):**
- MIR's virtual update uses a single **SGD** step (`theta - lr * grad`) rather than replicating the full
  Adam optimizer's moment state for a throwaway one-step lookahead — standard simplification in MIR
  reimplementations, since only the induced loss-increase ranking matters, not the exact update rule used
  to produce it.
- The interference candidate pool is the model's **entire** memory buffer (<=200 samples at this dataset's
  scale), not a further random subsample — one extra forward pass at this buffer size is cheap, so no
  additional approximation was needed specifically because of the small buffer.
- Both baselines use the **same total memory budget** (`memory_size`) as MUDVI, and reuse the identical
  model architecture, optimizer, batch sizes, epoch count, and fixed seed (`mudvi_baseline/run_comparison.py`),
  per the fair-comparison protocol Prof. Duan's feedback implies. The only architectural difference from
  MUDVI's own baseline is the memory buffer's partitioning policy (global reservoir vs. MUDVI's per-class
  reservoir, see `memory.py`'s own docstring) — the minimum change needed to make ER/MIR recognizable as
  the standard literature baselines rather than a relabeled copy of MUDVI's baseline.
- Per the explicit computational constraint for this comparison, only subjects A01–A03 are used (never
  A04–A09) **in `run_comparison.py`**; see that file's hardcoded `SUBJECTS` constant. The newer
  `run_experiment.py`/`gmed_baseline.py` path (added for the Lightning AI migration, see
  `LIGHTNING_MIGRATION.md`) is not subject to that constraint and runs any `--subjects` list, including
  the full A01–A09.
- GMED's memory-editing gradient step is likewise a single SGD-style step (one gradient-ascent update in
  INPUT space, not model-parameter space), evaluated with the model in `eval()` mode so BatchNorm's
  train()-mode batch statistics don't couple different memory examples' per-example gradients together —
  see `gmed_baseline.py`'s module docstring for the full list of documented simplifications relative to
  Jin et al. 2020's original algorithm.
