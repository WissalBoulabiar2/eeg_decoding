# MUDVI implementation fidelity: exact comparison against Duan et al.

This document is the direct answer to "before proceeding, compare our
implementation against the original MUDVI paper/code." Every claim below
was checked against `duanWLDAYT24nn.pdf` (Duan, Wang, Li, Doretto, Adjeroh,
Yin, Tao — "Online Continual Decoding of Streaming EEG Signal with a
Balanced and Informative Memory Buffer", preprint submitted to Elsevier;
MUDVI = **M**emory **U**pdate on **D**ata **V**olume and **I**nformativeness,
spelled out on p.7) by reading the relevant sections directly, not from
memory or from `IMPLEMENTATION_MAPPING.md` alone (that file was produced
before this re-check and is consistent with it, but this document is the
authoritative, re-verified version). No official code release for Duan et
al. is publicly available, so "vs. original code" below means "vs. the
paper's stated method" only.

## 1. Reproduced exactly

| Element | Paper location | Our code | Verified |
|---|---|---|---|
| Data segmentation: 400×22 segments, stride 50, window t=3–6s post-cue, 8 segments/trial | §4.3 "Data Processing", BCI IV-2a bullet | `data.py: SEGMENT_LEN=400, SEGMENT_STRIDE=50, WINDOW_START_S=3.0, WINDOW_END_S=6.0` | Exact match |
| Model family: 3-layer CNN (temporal conv filter (1,C) → depthwise conv filter (2,32) → pointwise conv), zero-padding between layers | §4.3 "Model Settings" (verbatim quote reproduced in `model.py`'s own docstring) | `model.py: MudviCNN` | Exact match on filter *shapes*; see §2 below for what is NOT given (channel counts) |
| BWT definition: BWT = 1/(N−1) · Σ_{i=1}^{N−1} (a_{N,i} − a_{i,i}) | §4.2, unnumbered equation | `metrics.py: compute_bwt` | Exact match, including the N−1 (not N) denominator |
| "Accuracy evaluated after sequential learning ends" as the headline metric | §4.2 | `metrics.py: evaluate`, `run_experiment.py`'s `final_avg_acc` | Exact match in definition |
| Memory update math, Eq. 1–4 (cluster replace probability A_C, move-in probability P_t^in, sampling probability q_t^C) | §3.1–3.2 | Implemented verbatim in `IMPLEMENTATION_MAPPING.md`'s trace table, **but see §3 below — this exact math is NOT what `memory.py` actually runs** | Equations transcribed correctly; **not** the code path in use (deliberate simplification, disclosed) |
| MMD kernel-shift-detection math, Eq. 8–12 (U-statistic MMD², Gaussian RKHS kernel, EMA-adaptive threshold μ_t/σ_t/a=1.96) | §3.3 | Equations transcribed correctly in `IMPLEMENTATION_MAPPING.md`; **not implemented as executable code — see §3** | Equations correct; implementation status is "not built", not "simplified" |
| λ=1 (memory move-in hyperparameter), R=6 (cluster-importance representatives), memory size 200 as a paper default | §4.3 "Model Settings", §4.6 | Memory size 200 used as our default (`config.py`); λ and R are moot since Eq. 1–4/the cluster mechanism itself is not the code path in use | N/A — see §3 |

## 2. Adapted (paper under-specifies, we made an explicit, documented choice)

| Gap in the paper | Our choice | Why |
|---|---|---|
| Layer channel counts: paper gives filter *shapes* (1,C) and (2,32) but never states the number of temporal filters (F1) | F1=16, depth multiplier=2 → 16×2=32, which reproduces the paper's own "(2,32)" literally | The one degree of freedom consistent with the paper's own stated number; no other value reproduces "32" as an output channel count from a stated depth-multiplier structure |
| Test protocol: paper's Table 1 numbers presumably use the dataset's standard train-session/eval-session split (BCI IV-2a ships `A0kT` for training and `A0kE`, with separate `.mat` label files, for evaluation) | We hold out a stratified 20% of **`A0kT`** trials as a permanent per-subject test set, and never use `A0kE` | The `A0kE` files present in this dataset folder carry event code `783` ("unknown") with no accompanying label file — there is no usable ground truth for them here. This is a real, disclosed data-availability gap, not a methodological preference — **see §4, this is likely the single largest driver of the absolute-accuracy gap documented there** |
| Training length (epochs/subject), batch size, learning rate for BCI IV-2a specifically | 15 epochs/subject, batch 32/32, Adam lr=1e-3 (`config.py` defaults) | Not stated anywhere in the paper for BCI IV-2a; reasonable defaults, exposed as CLI flags precisely so they can be swept (Table 3 in §5 below) rather than hardcoded |
| Reported statistics: paper reports mean±SD over **10 independent repeated runs** for every table (Fig. 5 caption: "each imbalance setting is repeatedly run for 10 times"; shift-detection ablation: "running... for 10 times on each dataset") | We report single-seed point estimates | Disclosed already in `paper/main.tex`'s Limitations section ("Single seed... due to the computational budget available"). This is a real fidelity gap for anyone reading our numbers as directly comparable to Table 1/3/4 of the original paper — ours are one draw, theirs are a mean of 10 |

## 3. Simplified (paper's mechanism not implemented; a documented, weaker substitute stands in for it)

These are the two most consequential differences, and neither is a matter
of degree — the paper's actual mechanism does not run in our code at all.

**3a. Memory update mechanism.** The paper's core contribution (Eq. 1–4,
Algorithm 1) is a hierarchical, per-**subject**-cluster buffer where (i)
which sample gets evicted depends on a gradient-norm *informativeness*
score and how over-represented that subject's cluster currently is
(Eq. 1–2), and (ii) which sample gets *replayed* is drawn with
informativeness-weighted probability, not uniformly (Eq. 4). **Our
`ClassBalancedMemory` (`memory.py`) instead partitions the buffer evenly
across the 4 fixed MI **classes** (not subjects) and uses plain uniform
reservoir sampling within each class partition — no informativeness score
is computed anywhere in our memory code.** This is disclosed prominently
in `memory.py`'s own docstring and in `IMPLEMENTATION_MAPPING.md` §1, but
it means: **what we call "MUDVI" in every result table in this repo is
not running Duan et al.'s memory algorithm.** It is a simpler
class-balanced-reservoir baseline that happens to sit in the same
trainer loop as our two proposed additions. Framing this as literally
"MUDVI" in a paper table without this caveat attached would overstate
fidelity.

**3b. Subject-shift detection.** The paper's kernel-MMD detector
(Eq. 8–12) is what makes the framework work in the realistic
*subject-agnostic* setting (subject identity unknown to the decoder) —
it is the mechanism that creates new memory clusters at the right time.
**We do not implement it.** `shift_detection.OracleShiftDetector`
substitutes a trivial oracle that reads the ground-truth subject id
directly. This is explicit in that module's own docstring: "this
baseline is explicitly subject-aware... the real MMD detector is out of
scope." Our own Relationship-Shift Detection (RSD) addition is a
*different*, independent signal (confidence/loss-based, not
feature-MMD-based) layered on top of this oracle — it does not replace
the missing MMD detector, and RSD's own accuracy-neutral, diagnostic-only
status (see `RSD_VALIDATION.md`) means the subject-agnostic capability
the original paper is built around is simply not present in this
codebase in any form.

**Net effect:** our "MUDVI baseline" is best described as *a MUDVI-style
sequential replay baseline built for BCI IV-2a, using a simplified
class-balanced memory and an oracle shift signal, evaluated under a
protocol adapted for this dataset folder's label availability* — not a
reproduction of Duan et al.'s memory-update algorithm. Every mention of
"MUDVI" in results tables going forward should either use this longer
description once, or a footnote pointing here.

## 4. The absolute-accuracy gap — read this before writing any comparison table

Table 1 of Duan et al. reports, for BCI IV-2a, **MUDVI: 45.98±1.83%
(imbalanced) / 50.24±1.67% (balanced)**, and their own ER/MIR baselines at
**39.24±1.35% / 42.53±1.15%** (Table 4, "Sequential" ordering — the same
ordering convention we use). Our from-scratch numbers, at the same
task (4-way BCI IV-2a MI classification, same architecture family), run
**substantially lower across every method**:

| Method | Duan et al. Table 1/4 (BCI IV-2a, sequential order) | This repo |
|---|---|---|
| MUDVI-style | 45.98% (imbalanced) / 50.24% (balanced) | ~29–31% (9-subject, seed 0) |
| ER | 39.24% | 31.1% (3-subject, seed 0) |
| MIR | 42.53% | 28.4% (3-subject, seed 0) |

This is a genuine, unresolved fidelity gap, not a rounding difference, and
it is **not** something to silently tune away. Plausible, currently
**unverified** contributors, roughly in order of suspected impact:

1. **Test-set protocol (§2 above).** We evaluate on a 20% held-out slice
   of the *same* recording session (`A0kT`) the model trained on; Duan et
   al. almost certainly evaluate against the dataset's dedicated
   evaluation-session recordings (`A0kE` + official labels), which is a
   materially different and likely harder generalization test (different
   day/session for the same subject is known to shift EEG statistics).
   If our 20%-holdout is *easier* than a genuine cross-session test, that
   would push our numbers **up** relative to theirs, not down — so this
   alone does not explain the gap in the observed direction, and is
   flagged as inconclusive rather than asserted.
2. **Number of repeated runs (§2).** Their numbers are a mean of 10 runs;
   ours are one seed. A single unlucky seed does not plausibly explain a
   ~15–20 point gap by itself, but it is a real, uncontrolled source of
   variance we have not yet measured (no multi-seed run exists anywhere
   in this repo for the MUDVI-style baseline).
3. **Architecture channel count (F1=16, §2).** An under- or
   over-provisioned first layer could plausibly cost double-digit
   accuracy on a 4-way task; this is untested (no F1 sweep exists).
4. **Training budget** (epochs/subject, batch size, lr) — entirely
   unspecified by the paper for BCI IV-2a; ours could simply be
   under-trained relative to whatever Duan et al. actually used.
5. **Missing informativeness-weighted memory (§3a).** If informativeness
   weighting materially improves which samples get retained/replayed,
   losing it would cost real accuracy — plausible, but we have no ablation
   isolating this from the other factors yet.

**Action before any comparison table is finalized:** this gap needs to be
either explained by a controlled experiment (e.g., an F1 sweep, a
multi-seed run) or explicitly reported as an open discrepancy in any
paper/report that uses these numbers. Do not present our ~29–31% baseline
next to Duan et al.'s ~46–50% Table 1 number as if they measure the same
thing — they are numbers from two different, incompletely-reconciled
protocols. Everywhere in this repo, comparisons should stay **within**
our own reimplementation (our MUDVI vs. our ER/MIR/GMED, all under the
identical protocol we control), which is internally fair even though it
is not yet externally validated against the published numbers.
