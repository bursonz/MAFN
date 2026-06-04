# Author: Boya Zhang <by.zhang1@siat.ac.cn>
# Date: 2026.05.31
"""Classification metrics (no sklearn dependency); all return plain floats."""
from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def _f1_per_class(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    p = tp / np.maximum(tp + fp, 1e-12)
    r = tp / np.maximum(tp + fn, 1e-12)
    return 2 * p * r / np.maximum(p + r, 1e-12)


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    return float(_f1_per_class(confusion_matrix(y_true, y_pred, num_classes)).mean())


def f1_weighted(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    weights = cm.sum(axis=1) / max(cm.sum(), 1)
    return float((_f1_per_class(cm) * weights).sum())


def sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return float(tp / max(tp + fn, 1)), float(tn / max(tn + fp, 1))


def precision_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    return float(tp / (tp + fp)) if tp + fp else float('nan')


def f1_positive(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    p = tp / max(tp + fp, 1e-12)
    r = tp / max(tp + fn, 1e-12)
    return float(2 * p * r / (p + r)) if (p + r) else 0.0


def auroc_binary(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUROC via the Mann-Whitney rank-sum. y_true in {0, 1}."""
    order = np.argsort(scores)
    y = y_true[order]
    ranks = np.arange(1, len(y) + 1)
    pos = ranks[y == 1].sum()
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    return float((pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Operating-point threshold maximising Youden's J (sensitivity + specificity - 1).

    A standard choice for imbalanced binary detection; used to report the fall
    detection operating point.
    """
    y_true = np.asarray(y_true).ravel()
    scores = np.asarray(scores).ravel()
    if (y_true == 1).sum() == 0 or (y_true == 0).sum() == 0:
        return 0.5
    thresholds = np.unique(scores)
    thresholds = np.concatenate(([thresholds[0] - 1e-6], thresholds))
    best_j, best_t = -2.0, 0.5
    for t in thresholds:
        pred = (scores >= t).astype(np.int64)
        sens, spec = sensitivity_specificity(y_true, pred)
        j = sens + spec - 1.0
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t
