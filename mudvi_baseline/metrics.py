"""Evaluation metrics, per Duan et al. Sec 4.2.

Acc(j, i): accuracy evaluated on subject i's test set, right after the
model finished training on subject j (j >= i).
BWT = mean_i [ a_{N,i} - a_{i,i} ], N = number of subjects. Negative BWT
means catastrophic forgetting; positive means later learning improved
earlier subjects.
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def evaluate(model, X: np.ndarray, y: np.ndarray, device, batch_size: int = 128) -> float:
    if len(y) == 0:
        return float("nan")
    model.eval()
    correct = 0
    for start in range(0, len(y), batch_size):
        xb = torch.from_numpy(X[start:start + batch_size]).to(device)
        yb = torch.from_numpy(y[start:start + batch_size]).to(device)
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
    return correct / len(y)


def compute_bwt(acc_matrix: dict, subject_order: list) -> float:
    """acc_matrix[step][subject_id] = accuracy on subject_id's test set right
    after training step `step` (0-indexed arrival order). subject_order[step]
    gives the subject id trained at that step, so Acc(i,i) is
    acc_matrix[i][subject_order[i]]."""
    last = max(acc_matrix)
    diffs = []
    for step, sid in enumerate(subject_order):
        if step == last:
            continue
        a_Ni = acc_matrix[last][sid]
        a_ii = acc_matrix[step][sid]
        diffs.append(a_Ni - a_ii)
    return float(np.mean(diffs)) if diffs else float("nan")


def compute_forgetting(acc_matrix: dict, subject_order: list) -> float:
    """Standard forgetting metric (Chaudhry et al. 2018, "Riemannian Walk
    for Incremental Learning"): for each subject i seen before the final
    step, f_i = max_{l in [i, N-1]} a_{l,i} - a_{N,i} -- the biggest drop
    between subject i's best-ever accuracy (usually right after it was
    trained, a_{i,i}, but the max is taken over the whole later trajectory
    in case accuracy on i improves again before declining) and its
    accuracy at the very end. Averaged over i < N-1 (the last-trained
    subject has no "later" steps to forget across). Unlike BWT, forgetting
    is always >= 0 by construction and only measures degradation, not
    backward transfer improvements."""
    last = max(acc_matrix)
    diffs = []
    for step, sid in enumerate(subject_order):
        if step == last:
            continue
        best = max(acc_matrix[l][sid] for l in range(step, last + 1))
        a_Ni = acc_matrix[last][sid]
        diffs.append(best - a_Ni)
    return float(np.mean(diffs)) if diffs else float("nan")
