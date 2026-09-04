"""High Gamma Dataset loading, trial extraction and segmentation.

The High Gamma Dataset (Schirrmeister et al. 2017, "Deep learning with
convolutional neural networks for EEG decoding and visualization") is
NOT one of the three datasets used in Duan et al. (BCI IV-2a, DEAP,
SEED -- see duanWLDAYT24nn.pdf Sec 4.3). Everything in this module is
therefore an IMPLEMENTATION DECISION, not a paper-derived fact -- it
extends the project beyond Duan et al. onto a fourth dataset, following
the SAME processing philosophy the paper applies to its own three
datasets (fixed-size, fixed-stride segments of raw, per-segment
z-normalized signal fed to the same architecture) rather than
reproducing the original HGD paper's own preprocessing (bandpass/
exponential-standardization/crop-training pipeline in braindecode's
`example.py`, which trains a different model with a different protocol
entirely).

File format (confirmed against braindecode's legacy
`braindecode.datasets.bbci.BBCIDataset`, the loader the HGD README
itself points to): each `<subject>.mat` is a MATLAB v7.3 (HDF5) file
with:
  - `nfo/T`      : scalar, total number of samples in the recording.
  - `nfo/fs`     : scalar, native sampling rate in Hz (500 for this
                   dataset).
  - `nfo/clab`   : one HDF5 object reference per channel, each pointing
                   to that channel's name as an array of char codes.
  - `nfo/className`: same encoding, one object reference per class name.
  - `ch1`, `ch2`, ... : one continuous 1-D signal vector per channel,
                   1-indexed to match MATLAB/`clab` ordering.
  - `mrk/time`   : event (cue) onset times, in milliseconds.
  - `mrk/event/desc`: integer class code per event, 1-indexed in the
                   same order as `nfo/className`.

Channel selection: only channels not prefixed `BIP`/`E`/`Microphone`/
`Breath`/`GSR` are used, identical to `BBCIDataset._determine_sensors`'s
default (all-sensors) behaviour -- this yields all 128 EEG channels
for this dataset (asserted below). This is the dataset's native full
EEG montage, not the ~44-channel motor-cortex subset the original HGD
paper's own `example.py` further restricts to -- IMPLEMENTATION
DECISION: keep the full montage since neither Duan et al. nor this
project specifies a channel-selection recipe of its own, and
`MudviCNN`'s channel count is already a free hyperparameter per
dataset (see `model.py`; Duan et al. themselves use C=22/32/62 for
their three datasets).

Class codes: for the standard 4-class recordings used here, event
codes 1..4 map to class names `["Right Hand", "Left Hand", "Rest",
"Feet"]` in that order (verified against
`BBCIDataset._check_class_names`'s primary branch and `example.py`'s
`marker_def`). Labels below are 0-indexed as `code - 1`.

Segmentation: resampled to 250 Hz (matching BCI IV-2a's rate exactly)
via `scipy.signal.resample_poly`, then segmented with the SAME
`SEGMENT_LEN`/`SEGMENT_STRIDE` constants as `data.py` (400 x C, stride
50) over a 3-second post-cue window (t=0s to t=3s relative to the cue
event) -- IMPLEMENTATION DECISION chosen to reproduce BCI IV-2a's own
3-second window length and 8-segments-per-trial count exactly (poor
man's cross-dataset comparability), since HGD's cue marks movement
onset directly (no separate trial-start/cue-onset pair like BCI IV-2a's
768->769..772), so there is no equivalent "t=3s post trial start"
reference point to reuse verbatim.

Train/test split: HGD ships pre-split `train/<subject>.mat` and
`test/<subject>.mat` files per subject (the dataset's own official
split), used directly here instead of re-deriving one from trials
(unlike `data.py`, which has to invent a split because BCI IV-2a's
`.gdf` test files carry no usable labels -- see that module's
docstring).
"""
from __future__ import annotations

import os
from fractions import Fraction

import h5py
import numpy as np
from scipy.signal import resample_poly

from .data import SubjectData, _segment_trial, SEGMENT_LEN, SEGMENT_STRIDE

TARGET_SFREQ = 250.0
WINDOW_START_S = 0.0
WINDOW_END_S = 3.0
NUM_CLASSES = 4
CLASS_NAMES = ["Right Hand", "Left Hand", "Rest", "Feet"]

_EXCLUDED_PREFIXES = ("BIP", "E", "Microphone", "Breath", "GSR")


def _decode_char_ref(h5file: "h5py.File", ref) -> str:
    return "".join(chr(c) for c in h5file[ref][:].squeeze())


def _get_all_sensor_names(h5file: "h5py.File") -> list[str]:
    clab_refs = h5file["nfo"]["clab"][:].squeeze()
    return [_decode_char_ref(h5file, ref) for ref in clab_refs]


def _eeg_channel_indices(all_sensor_names: list[str]) -> list[int]:
    indices = [
        i for i, name in enumerate(all_sensor_names)
        if not name.startswith(_EXCLUDED_PREFIXES)
    ]
    assert len(indices) in (16, 32, 64, 128), (
        f"Unexpected EEG channel count {len(indices)}; recheck channel "
        f"filtering against the actual file contents."
    )
    return indices


def _load_bbci_mat(path: str):
    """Returns (signal: (n_channels, n_samples) float32 uV, fs: float,
    event_samples: (n_events,) int, event_labels: (n_events,) int 0..3)."""
    with h5py.File(path, "r") as h5file:
        fs = float(h5file["nfo"]["fs"][0, 0])
        n_samples = int(h5file["nfo"]["T"][0, 0])

        all_sensor_names = _get_all_sensor_names(h5file)
        chan_indices = _eeg_channel_indices(all_sensor_names)

        signal = np.empty((len(chan_indices), n_samples), dtype=np.float32)
        for out_i, chan_i in enumerate(chan_indices):
            chan_name = f"ch{chan_i + 1}"  # matlab/hdf5 1-based naming
            signal[out_i, :] = h5file[chan_name][:].squeeze()

        event_times_ms = h5file["mrk"]["time"][:].squeeze()
        event_codes = h5file["mrk"]["event"]["desc"][:].squeeze().astype(np.int64)

        class_name_refs = h5file["nfo"]["className"][:].squeeze()
        class_names = [_decode_char_ref(h5file, ref) for ref in class_name_refs]

    assert class_names == CLASS_NAMES, (
        f"Unexpected class name order {class_names} in {path}; the "
        f"code-1 -> label mapping below assumes {CLASS_NAMES}."
    )

    event_samples = np.round(event_times_ms * fs / 1000.0).astype(np.int64)
    event_labels = event_codes - 1
    return signal, fs, event_samples, event_labels


def _resample_to_target(signal: np.ndarray, fs: float) -> tuple[np.ndarray, float]:
    """signal: (n_channels, n_samples) -> resampled to TARGET_SFREQ Hz."""
    if fs == TARGET_SFREQ:
        return signal, fs
    ratio = Fraction(TARGET_SFREQ / fs).limit_denominator(1000)
    resampled = resample_poly(signal, ratio.numerator, ratio.denominator, axis=1)
    return resampled.astype(np.float32), TARGET_SFREQ


def _load_split(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Loads one train/ or test/ .mat file and returns stacked
    (X, y) segments/labels across all trials in that file."""
    signal, native_fs, event_samples, event_labels = _load_bbci_mat(path)
    signal, resampled_fs = _resample_to_target(signal, native_fs)
    # rescale event onsets from the native timeline to the resampled one
    scale = resampled_fs / native_fs
    event_samples = np.round(event_samples * scale).astype(np.int64)

    win_start = int(round(WINDOW_START_S * resampled_fs))
    win_end = int(round(WINDOW_END_S * resampled_fs))

    all_segments = []
    all_labels = []
    for onset, label in zip(event_samples, event_labels):
        lo = onset + win_start
        hi = onset + win_end
        if lo < 0 or hi > signal.shape[1]:
            continue
        trial_signal = signal[:, lo:hi].T  # (n_samples, C)
        segs = _segment_trial(trial_signal)  # (n_segments, C, SEGMENT_LEN)
        all_segments.append(segs)
        all_labels.append(np.full(segs.shape[0], label, dtype=np.int64))

    if not all_segments:
        n_channels = signal.shape[0]
        return (np.zeros((0, n_channels, SEGMENT_LEN), dtype=np.float32),
                np.zeros((0,), dtype=np.int64))
    X = np.concatenate(all_segments, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y


def load_subject(subject_id: str, data_dir: str, **_ignored) -> SubjectData:
    """Loads one High Gamma Dataset subject.

    `data_dir` is the HGD `data/` folder containing `train/` and
    `test/` subfolders (e.g. the `data` subfolder of a checkout of
    https://web.gin.g-node.org/robintibor/high-gamma-dataset).
    `subject_id` is the numeric subject id as used in the HGD filenames
    (`"1"`.."14"`, not zero-padded like BCI IV-2a's `"01"`).

    Accepts and ignores `test_fraction`/`seed`/`drop_artifact_trials`
    kwargs so it is a drop-in replacement for `data.load_subject` in
    `run_experiment.py` -- HGD's train/test split comes from the
    dataset's own files, not a synthesized one.
    """
    subject_num = str(int(subject_id))
    train_path = os.path.join(data_dir, "train", f"{subject_num}.mat")
    test_path = os.path.join(data_dir, "test", f"{subject_num}.mat")

    X_train, y_train = _load_split(train_path)
    X_test, y_test = _load_split(test_path)

    return SubjectData(
        subject_id=subject_id,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
    )
