"""BCI Competition IV 2a loading, trial extraction and segmentation.

Facts below were confirmed by direct inspection of the GDF files with
mne 1.12.1 (see IMPLEMENTATION_MAPPING.md Sec 0), not assumed:
  - 25 recorded channels: 22 EEG + 3 EOG (EOG dropped, not used).
  - 250 Hz sampling rate.
  - Event 768 = trial start (t=0 reference). The matching cue event
    (769=left hand, 770=right hand, 771=both feet, 772=tongue) follows
    768 by exactly 500 samples (2s), matching the official BCI IV 2a
    timing protocol (cue at t=2s).
  - Only `A0kT.gdf` files carry real labels. `A0kE.gdf` files only carry
    event 783 ("unknown") -- no usable labels are present in this data
    folder, so E files are not used anywhere in this pipeline
    (IMPLEMENTATION DECISION, matches IMPLEMENTATION_MAPPING.md Sec 0).

Segmentation follows Duan et al. Sec 4.3 "Data Processing" (BCI IV-2a):
  - Each trial is split into segments of size 400 (time) x 22 (channels).
  - Stride 50 between adjacent segments.
  - The window t=3s to t=6s post trial-start is used -> 750 samples
    (250 Hz) -> (750-400)/50 + 1 = 8 segments per trial.

Bandpass filtering is NOT applied (not specified in Duan et al. Sec 4.3
for BCI IV-2a; IMPLEMENTATION DECISION). Each segment is z-normalized
per channel across time, which is the minimal normalization needed to
train a CNN and is applied uniformly regardless of downstream use.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import mne

mne.set_log_level("ERROR")

EEG_CHANNELS = [
    "EEG-Fz", "EEG-0", "EEG-1", "EEG-2", "EEG-3", "EEG-4", "EEG-5",
    "EEG-C3", "EEG-6", "EEG-Cz", "EEG-7", "EEG-C4", "EEG-8", "EEG-9",
    "EEG-10", "EEG-11", "EEG-12", "EEG-13", "EEG-14", "EEG-Pz", "EEG-15",
    "EEG-16",
]
assert len(EEG_CHANNELS) == 22

SFREQ = 250.0
TRIAL_START_CODE = "768"
CUE_CODES = {"769": 0, "770": 1, "771": 2, "772": 3}  # left, right, feet, tongue
ARTIFACT_CODE = "1023"

SEGMENT_LEN = 400
SEGMENT_STRIDE = 50
WINDOW_START_S = 3.0
WINDOW_END_S = 6.0
NUM_CLASSES = 4


@dataclass
class SubjectData:
    subject_id: str
    X_train: np.ndarray  # (n_segments, 22, 400)
    y_train: np.ndarray  # (n_segments,)
    X_test: np.ndarray
    y_test: np.ndarray


def _extract_trials(raw: "mne.io.Raw", drop_artifact_trials: bool = False):
    """Return list of (start_sample, label) for each usable trial.

    drop_artifact_trials: opt-in, default OFF. Duan et al. never mention
    artifact rejection for BCI IV-2a; kept off by default to stay
    maximally faithful to the paper (IMPLEMENTATION DECISION, matches
    IMPLEMENTATION_MAPPING.md Sec 0).
    """
    events, event_id = mne.events_from_annotations(raw)
    inv = {v: k for k, v in event_id.items()}

    artifact_samples = set()
    if drop_artifact_trials and ARTIFACT_CODE in event_id:
        artifact_samples = {
            e[0] for e in events if inv[e[2]] == ARTIFACT_CODE
        }

    trials = []
    n = len(events)
    for i, e in enumerate(events):
        code = inv[e[2]]
        if code != TRIAL_START_CODE:
            continue
        start_sample = e[0]
        if drop_artifact_trials and start_sample in artifact_samples:
            continue
        # the cue code is the very next event after trial start
        if i + 1 >= n:
            continue
        next_code = inv[events[i + 1][2]]
        if next_code not in CUE_CODES:
            continue
        trials.append((start_sample, CUE_CODES[next_code]))
    return trials


def _segment_trial(trial_signal: np.ndarray) -> np.ndarray:
    """trial_signal: (750, 22) -> (8, 22, 400), per-channel z-normalized."""
    n_samples = trial_signal.shape[0]
    segments = []
    start = 0
    while start + SEGMENT_LEN <= n_samples:
        seg = trial_signal[start:start + SEGMENT_LEN, :]  # (400, 22)
        seg = seg.T  # (22, 400)
        mu = seg.mean(axis=1, keepdims=True)
        sd = seg.std(axis=1, keepdims=True) + 1e-8
        seg = (seg - mu) / sd
        segments.append(seg.astype(np.float32))
        start += SEGMENT_STRIDE
    return np.stack(segments, axis=0)


def load_subject_gdf(subject_id: str, data_dir: str,
                      drop_artifact_trials: bool = False):
    """Load `A0{subject}T.gdf`, return per-trial segments and labels.

    Returns:
      trial_segments: list of (8, 22, 400) arrays, one per trial
      trial_labels: list of int labels (0..3), one per trial
    """
    path = os.path.join(data_dir, f"A{subject_id}T.gdf")
    raw = mne.io.read_raw_gdf(path, preload=True, eog=["EOG-left", "EOG-central", "EOG-right"])
    raw.pick(EEG_CHANNELS)

    trials = _extract_trials(raw, drop_artifact_trials=drop_artifact_trials)

    data = raw.get_data()  # (22, n_total_samples), volts
    data = data * 1e6  # convert to microvolts for a friendlier numeric scale

    win_start = int(round(WINDOW_START_S * SFREQ))
    win_end = int(round(WINDOW_END_S * SFREQ))

    trial_segments = []
    trial_labels = []
    for start_sample, label in trials:
        lo = start_sample + win_start
        hi = start_sample + win_end
        if hi > data.shape[1]:
            continue
        trial_signal = data[:, lo:hi].T  # (750, 22)
        segs = _segment_trial(trial_signal)  # (8, 22, 400)
        trial_segments.append(segs)
        trial_labels.append(label)

    return trial_segments, trial_labels


def load_subject(subject_id: str, data_dir: str, test_fraction: float = 0.2,
                  seed: int = 0, drop_artifact_trials: bool = False) -> SubjectData:
    """Load one subject and split into train/test at the TRIAL level.

    Splitting is done on trials (not on the 8 overlapping segments a
    trial produces) to avoid leaking near-identical windows between
    train and test, stratified by class. This mirrors the ~20%-test
    split used elsewhere in Duan et al. (IMPLEMENTATION DECISION, no
    E-file labels are available locally -- see module docstring).
    """
    trial_segments, trial_labels = load_subject_gdf(
        subject_id, data_dir, drop_artifact_trials=drop_artifact_trials
    )
    trial_labels = np.array(trial_labels)
    rng = np.random.RandomState(seed)

    train_idx, test_idx = [], []
    for c in range(NUM_CLASSES):
        idx = np.where(trial_labels == c)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_fraction)))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())

    def _stack(indices):
        if not indices:
            return (np.zeros((0, 22, SEGMENT_LEN), dtype=np.float32),
                     np.zeros((0,), dtype=np.int64))
        X = np.concatenate([trial_segments[i] for i in indices], axis=0)
        y = np.concatenate(
            [np.full(trial_segments[i].shape[0], trial_labels[i], dtype=np.int64)
             for i in indices], axis=0
        )
        return X, y

    X_train, y_train = _stack(train_idx)
    X_test, y_test = _stack(test_idx)

    return SubjectData(
        subject_id=subject_id,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
    )
