"""NOT/GLR change-point detector (mudvi_gp_report.pdf Sec 3.3 Eq. 8),
applied to a scalar stream (Addition 2's confidence/loss signal, see
`confidence_detector.py`). Generic: does not know anything about EEG or
the model, only consumes a scalar stream.

IMPLEMENTATION DECISION (documented in IMPLEMENTATION_MAPPING.md Sec 3):
the report gives the GLR statistic itself (Eq. 8) but not the surrounding
operational parameters (window size, candidate split points, threshold).
This module:
  - maintains a sliding window of the last `window_size` values (default
    40, configurable);
  - at each step, fits a single OLS line to the whole window (null
    hypothesis, "one-line fit") vs. two independent OLS lines split at
    an interior point c (alternative, "two-line fit"), Gaussian
    likelihood, and takes the maximum GLR statistic over all candidate c;
  - flags a change-point if that maximum exceeds a threshold calibrated
    by Monte Carlo simulation (see `calibrate_threshold()` below).
This is standard NOT/GLR practice (Baranowski, Chen & Fryzlewicz 2019),
applied here because the report gives the statistic but not these
operational parameters -- not attributed to either paper as a given.

CORRECTNESS-CRITICAL DECISION -- `min_segment_length`: candidate split
points c are restricted to c in [min_segment_length, n-min_segment_length]
(default min_segment_length=5), NOT the full [2, n-2] range naively
implied by "every interior point". An OLS line has exactly 2 free
parameters, so a line fit to exactly 2 points is a PERFECT fit (residual
sum of squares = 0) for ANY data, purely by construction -- with no
minimum segment length, the c=2 (or symmetric n-2) edge case always wins
the max-over-c search and produces an enormous, data-independent
statistic value, causing the detector to flag a change-point on ~100% of
windows regardless of whether the stream actually shifted (verified
empirically: 200/200 false positives on pure noise before this fix was
added). min_segment_length=5 keeps enough points per segment that a
perfect/near-perfect fit is no longer a trivial degenerate case.

CALIBRATION DECISION -- Monte Carlo threshold, not a closed-form
chi-square quantile: `not_glr_statistic` reports the MAXIMUM GLR
statistic over ~n_candidates dependent split points, not a single test.
A naive single-test chi2(df=2) threshold under-covers this maximum
badly (measured: ~100% false-positive rate on pure noise). A Bonferroni-
corrected chi2 threshold (per-candidate significance level divided by
n_candidates) is a valid but loose upper bound AND additionally relies on
chi2(df=2) being an accurate approximation of the per-candidate null
distribution, which is only asymptotic (large-segment) and measurably
inaccurate for the small segments used here (measured: ~5.8% actual
false-positive rate vs. the nominal 1% target). `calibrate_threshold()`
instead simulates the exact statistic's null distribution directly via
Monte Carlo and takes its empirical quantile -- measured ~1.06% false
positives on held-out noise, matching the 1% target. This is exact
(subject to simulation count) rather than approximate, and only needs to
be run once per (window_size, min_segment_length) configuration since the
statistic is provably invariant to the stream's overall scale and to any
shared linear trend (both the null and alternative models fit their own
intercept+slope per segment, so scale/trend cancel exactly in the
log-ratio) -- so calibrating on standard N(0,1) noise gives a threshold
valid for a confidence/loss stream at any real scale.

REFINEMENT -- block-bootstrap calibration from a REAL null stream
(`calibrate_threshold_from_stream`): `calibrate_threshold` still assumes
the null (no-shift) distribution is i.i.d. Gaussian noise. The actual
confidence/loss stream produced during training is not i.i.d. -- it is
mildly autocorrelated (consecutive steps use a model that has only
changed a little, and EEG segments within/across mini-batches overlap),
and cross-entropy/confidence values are not exactly Gaussian (e.g.
bounded below at 0). If real "stable" stretches have more (or less)
apparent trend structure than synthetic noise, the true false-positive
rate on the real stream can differ from what synthetic-noise calibration
predicts. `calibrate_threshold_from_stream` fixes this by drawing
candidate windows via a MOVING BLOCK BOOTSTRAP directly from a real
observed stream that is genuinely shift-free by construction (see
`confidence_detector.compute_null_stream`: the frozen reference model
evaluated on the very data it was trained on) -- preserving whatever
real autocorrelation/shape the signal actually has, at the cost of
needing that real stream available before calibrating.
"""
from __future__ import annotations

import numpy as np


def _line_rss(y: np.ndarray) -> float:
    """Residual sum of squares of the best-fit OLS line y ~ a + b*t."""
    n = len(y)
    if n < 2:
        return 0.0
    t = np.arange(n, dtype=np.float64)
    coeffs = np.polyfit(t, y, deg=1)
    fitted = np.polyval(coeffs, t)
    return float(np.sum((y - fitted) ** 2))


def not_glr_statistic(window: np.ndarray, min_segment_length: int = 5):
    """Eq. 8: max over interior split points c (each segment at least
    `min_segment_length` points -- see module docstring for why this
    floor is required for correctness) of the Gaussian GLR statistic
    comparing a single-line fit over the whole window (null) to two
    independent line fits before/after c (alternative).

    Returns (best_statistic, best_split_index). If the window is too
    short to admit any valid split (n < 2*min_segment_length), returns
    (-inf, None) -- i.e. never flags a change-point.
    """
    n = len(window)
    rss0 = _line_rss(window)
    eps = 1e-12
    best_stat, best_c = -np.inf, None
    for c in range(min_segment_length, n - min_segment_length + 1):
        rss1 = _line_rss(window[:c])
        rss2 = _line_rss(window[c:])
        stat = n * np.log(rss0 / n + eps) - (
            c * np.log(rss1 / c + eps) + (n - c) * np.log(rss2 / (n - c) + eps)
        )
        if stat > best_stat:
            best_stat, best_c = stat, c
    return best_stat, best_c


def calibrate_threshold(window_size: int, min_segment_length: int = 5,
                         confidence_level: float = 0.99,
                         n_simulations: int = 8000, seed: int = 12345) -> float:
    """Monte Carlo calibration of the change-point threshold under the
    null (no real shift): simulate `n_simulations` windows of standard
    N(0,1) noise, compute `not_glr_statistic` for each, and return the
    empirical `confidence_level` quantile. Valid for a confidence/loss
    stream at ANY real scale/trend -- see module docstring ("CALIBRATION
    DECISION") for why calibrating on standard noise is sufficient."""
    rng = np.random.RandomState(seed)
    stats = np.empty(n_simulations)
    for i in range(n_simulations):
        window = rng.normal(size=window_size)
        stats[i], _ = not_glr_statistic(window, min_segment_length=min_segment_length)
    return float(np.quantile(stats, confidence_level))


def calibrate_threshold_from_stream(null_stream: np.ndarray, window_size: int,
                                     min_segment_length: int = 5,
                                     confidence_level: float = 0.99,
                                     n_simulations: int = 8000, seed: int = 12345) -> float:
    """Moving-block-bootstrap calibration (see module docstring,
    "REFINEMENT"): repeatedly draw a random contiguous window of length
    `window_size` from a REAL observed `null_stream` that is genuinely
    shift-free by construction, compute `not_glr_statistic` for each draw,
    and return the empirical `confidence_level` quantile. Unlike
    `calibrate_threshold`, this preserves the real signal's
    autocorrelation and non-Gaussian shape instead of assuming i.i.d.
    Gaussian noise."""
    n = len(null_stream)
    if n < window_size:
        raise ValueError(f"null_stream has only {n} points, need >= window_size ({window_size}).")
    rng = np.random.RandomState(seed)
    max_start = n - window_size
    stats = np.empty(n_simulations)
    for i in range(n_simulations):
        start = rng.randint(0, max_start + 1)
        window = null_stream[start:start + window_size]
        stats[i], _ = not_glr_statistic(window, min_segment_length=min_segment_length)
    return float(np.quantile(stats, confidence_level))


class ChangePointDetector:
    """Sliding-window NOT/GLR detector over a scalar stream."""

    def __init__(self, window_size: int = 40, confidence_level: float = 0.99,
                 min_segment_length: int = 5, n_calibration_simulations: int = 8000,
                 calibration_seed: int = 12345, null_stream: np.ndarray | None = None,
                 precomputed_threshold: float | None = None):
        if window_size < 2 * min_segment_length:
            raise ValueError(
                f"window_size ({window_size}) must be >= 2*min_segment_length "
                f"({2 * min_segment_length}) so at least one valid split exists."
            )
        self.window_size = window_size
        self.min_segment_length = min_segment_length
        # Calibrated threshold -- see the module docstring's "CALIBRATION
        # DECISION" and "REFINEMENT". If a real `null_stream` is provided
        # (the frozen reference model's own confidence/loss values on its
        # own training data -- see confidence_detector.py), calibrate
        # against it via block bootstrap (more faithful to the real
        # signal's autocorrelation/shape). Otherwise fall back to
        # synthetic i.i.d. Gaussian-noise calibration. One-time cost paid
        # once per ChangePointDetector construction (i.e. once per
        # training run), negligible next to the run itself.
        # `precomputed_threshold`, when given, skips recalibration entirely
        # (checkpoint-resume path: the threshold was already calibrated
        # once before the run was interrupted -- see `state_dict`/
        # `load_state_dict` below and checkpoint.py. Re-running Monte Carlo
        # calibration on resume would be wasteful and, for the
        # block-bootstrap variant, would require re-deriving `null_stream`,
        # which is not persisted).
        if precomputed_threshold is not None:
            self.threshold = precomputed_threshold
        elif null_stream is not None:
            self.threshold = calibrate_threshold_from_stream(
                null_stream, window_size=window_size, min_segment_length=min_segment_length,
                confidence_level=confidence_level, n_simulations=n_calibration_simulations,
                seed=calibration_seed,
            )
        else:
            self.threshold = calibrate_threshold(
                window_size=window_size, min_segment_length=min_segment_length,
                confidence_level=confidence_level, n_simulations=n_calibration_simulations,
                seed=calibration_seed,
            )
        self._buffer: list[float] = []

    def state_dict(self) -> dict:
        """Calibrated threshold + sliding-window contents, for
        checkpointing (avoids re-running Monte Carlo calibration on
        resume, and preserves the exact in-flight window)."""
        return {
            "window_size": self.window_size,
            "min_segment_length": self.min_segment_length,
            "threshold": self.threshold,
            "buffer": list(self._buffer),
        }

    def load_state_dict(self, state: dict) -> None:
        assert state["window_size"] == self.window_size
        assert state["min_segment_length"] == self.min_segment_length
        self.threshold = state["threshold"]
        self._buffer = list(state["buffer"])

    def update(self, value: float) -> bool:
        """Feed one new scalar value; return True if a change-point is
        flagged using the current sliding window (False until the window
        is full)."""
        self._buffer.append(float(value))
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)
        if len(self._buffer) < self.window_size:
            return False
        stat, _ = not_glr_statistic(np.asarray(self._buffer, dtype=np.float64),
                                     min_segment_length=self.min_segment_length)
        # bool(...) -- not_glr_statistic returns a numpy scalar, so the
        # naive comparison below is np.bool_, not a native Python bool.
        # That silently breaks json.dump on anything that stores this
        # value raw (e.g. run_experiment.py's per-step shift_log), which
        # is exactly the kind of latent type bug that only surfaces once
        # raw per-step diagnostics are actually persisted rather than
        # just summed.
        return bool(stat > self.threshold)
