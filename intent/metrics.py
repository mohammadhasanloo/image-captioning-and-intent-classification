"""Multi-class metrics accumulated over a whole split.

Counts are summed across every sample before any ratio is taken. Precision,
recall and F1 are ratios of counts, so averaging them over batches gives a
different and misleading number; only accuracy survives that treatment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MulticlassMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    micro_f1: float
    support: int

    def format(self) -> str:
        return (
            f"accuracy {self.accuracy:.4f}  macro-F1 {self.macro_f1:.4f}  "
            f"micro-F1 {self.micro_f1:.4f}  (n={self.support})"
        )


def _labels(array: np.ndarray) -> np.ndarray:
    """Accept either integer labels or one-hot rows."""
    array = np.asarray(array)
    return array.argmax(axis=1) if array.ndim == 2 and array.shape[1] > 1 else array.ravel()


def evaluate(y_true: np.ndarray, y_predicted: np.ndarray) -> MulticlassMetrics:
    """Score a whole split at once, macro and micro averaged."""
    true_labels, predicted_labels = _labels(y_true), _labels(y_predicted)
    if true_labels.shape != predicted_labels.shape:
        raise ValueError(f"shape mismatch: {true_labels.shape} against {predicted_labels.shape}")

    classes = np.unique(np.concatenate([true_labels, predicted_labels]))
    precisions, recalls, f1s = [], [], []
    total_tp = total_fp = total_fn = 0

    for label in classes:
        tp = int(np.sum((true_labels == label) & (predicted_labels == label)))
        fp = int(np.sum((true_labels != label) & (predicted_labels == label)))
        fn = int(np.sum((true_labels == label) & (predicted_labels != label)))
        total_tp, total_fp, total_fn = total_tp + tp, total_fp + fp, total_fn + fn

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)

    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )

    return MulticlassMetrics(
        accuracy=float(np.mean(true_labels == predicted_labels)),
        macro_precision=float(np.mean(precisions)),
        macro_recall=float(np.mean(recalls)),
        macro_f1=float(np.mean(f1s)),
        micro_f1=float(micro_f1),
        support=int(len(true_labels)),
    )
