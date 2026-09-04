"""High Gamma Dataset loader for the Kaggle-hosted, pre-processed `.fif`
export (Kaggle dataset `buzhichun/high-gamma-dataset-fif`), used instead
of `data_high_gamma.py`'s raw-`.mat`-from-GIN path when GIN itself is
unreachable (GIN blocks Kaggle's egress IP range -- confirmed by
comparing an identical request from a non-Kaggle network, which
succeeds, against the same request from Kaggle, which 403s).

Layout (confirmed by direct inspection on 2026-09-04, not assumed):
  <data_dir>/Subject <N>/<0|1>/description.json   -- {"subject", "session", "run"}
                          .../target_name.json     -- always {"target_name": null}
                                                       here: this is a raw
                                                       CONTINUOUS recording
                                                       (pre-windowing), so
                                                       labels live in the
                                                       recording's own
                                                       annotations, not a
                                                       separate per-window
                                                       target file.
                          .../raw_preproc_kwargs.json -- preprocessing already
                                                       applied before export:
                                                       resampled to 250Hz,
                                                       picked down to a fixed
                                                       50-channel montage.
                          .../<0|1>-raw.fif        -- the continuous MNE Raw

This is a `braindecode`/MOABB export of MOABB's `Schirrmeister2017`
dataset (github.com/NeuroTechX/moabb,
`moabb/datasets/schirrmeister2017.py`), which itself loads the same
official `.edf` files this project's other High Gamma path reads as
`.mat` from GIN -- same underlying recordings, different
serialization/preprocessing stage. MOABB's own `events` mapping (from
that file, verified, not guessed): `{"right_hand": 1, "left_hand": 2,
"rest": 3, "feet": 4}`; annotation description strings are matched
case/underscore-insensitively against that mapping rather than assumed
literally, since the exact string casing surviving a resample+pick_channels
round-trip through braindecode's own annotation handling was not
independently confirmed.

Reuses `data.py`'s `_segment_trial`/`SEGMENT_LEN`/`SEGMENT_STRIDE` and the
same 0-3s post-event window convention as `data_high_gamma.py`'s
`.mat`-from-GIN path (see that module's docstring for why: matching BCI
IV-2a's 3-second window/8-segments-per-trial exactly, chosen once and
reused here for consistency between the two High Gamma access paths
rather than re-deriving a second, different IMPLEMENTATION DECISION).
Already at 250Hz per `raw_preproc_kwargs.json`, so no resampling here.
"""
from __future__ import annotations

import glob
import json
import os

import mne
import numpy as np

from .data import SubjectData, SEGMENT_LEN, _segment_trial

mne.set_log_level("ERROR")

WINDOW_START_S = 0.0
WINDOW_END_S = 3.0
NUM_CLASSES = 4

_LABEL_MAP = {"right hand": 0, "left hand": 1, "rest": 2, "feet": 3}


def _normalize_label(desc: str) -> str:
    return desc.strip().lower().replace("_", " ")


def _find_run_dir(subject_dir: str, run_name: str) -> str:
    """Finds the subfolder whose description.json has "run": run_name,
    rather than assuming folder "0" is always train -- not guaranteed by
    the export, only observed for subject 1."""
    candidates = []
    for entry in sorted(os.listdir(subject_dir)):
        run_dir = os.path.join(subject_dir, entry)
        desc_path = os.path.join(run_dir, "description.json")
        if not os.path.isfile(desc_path):
            continue
        with open(desc_path) as f:
            desc = json.load(f)
        if desc.get("run") == run_name:
            candidates.append(run_dir)
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one run={run_name!r} subfolder under "
            f"{subject_dir} (found {len(candidates)}: {candidates}). "
            f"Check each subfolder's description.json."
        )
    return candidates[0]


def _find_raw_fif(run_dir: str) -> str:
    matches = glob.glob(os.path.join(run_dir, "*-raw.fif"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one *-raw.fif in {run_dir}, found {matches}")
    return matches[0]


def _load_run(run_dir: str):
    raw = mne.io.read_raw_fif(_find_raw_fif(run_dir), preload=True, verbose=False)
    sfreq = raw.info["sfreq"]
    data = raw.get_data() * 1e6  # volts -> microvolts, matches data.py's convention
    n_channels = data.shape[0]

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    inv = {v: k for k, v in event_id.items()}

    win_start = int(round(WINDOW_START_S * sfreq))
    win_end = int(round(WINDOW_END_S * sfreq))

    segments, labels = [], []
    for onset_sample, _, code in events:
        key = _normalize_label(inv[code])
        if key not in _LABEL_MAP:
            continue  # non-trial annotation (e.g. a boundary/bad-segment marker)
        lo, hi = onset_sample + win_start, onset_sample + win_end
        if lo < 0 or hi > data.shape[1]:
            continue
        trial_signal = data[:, lo:hi].T  # (n_samples, C)
        segs = _segment_trial(trial_signal)  # (n_segments, C, SEGMENT_LEN)
        segments.append(segs)
        labels.append(np.full(segs.shape[0], _LABEL_MAP[key], dtype=np.int64))

    if not segments:
        return (np.zeros((0, n_channels, SEGMENT_LEN), dtype=np.float32),
                np.zeros((0,), dtype=np.int64))
    return np.concatenate(segments, axis=0), np.concatenate(labels, axis=0)


def load_subject(subject_id: str, data_dir: str, **_ignored) -> SubjectData:
    """`data_dir` is the folder containing `Subject <N>/` subfolders
    (e.g. wherever `buzhichun/high-gamma-dataset-fif` was unzipped to).
    `subject_id` is the numeric subject id, `"1"`..`"14"`."""
    subject_dir = os.path.join(data_dir, f"Subject {int(subject_id)}")
    train_dir = _find_run_dir(subject_dir, "train")
    test_dir = _find_run_dir(subject_dir, "test")

    X_train, y_train = _load_run(train_dir)
    X_test, y_test = _load_run(test_dir)

    return SubjectData(
        subject_id=subject_id,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
    )
