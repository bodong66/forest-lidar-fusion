"""Evaluation metrics: confusion matrix, IoU, accuracy, F1."""

from __future__ import annotations

import numpy as np


class ConfusionMatrix:
    def __init__(self, n_classes: int, ignore_index: int = 255) -> None:
        self.n, self.ignore = n_classes, ignore_index
        self.mat = np.zeros((n_classes, n_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        pred, target = np.asarray(pred).ravel(), np.asarray(target).ravel()
        keep = target != self.ignore
        idx = target[keep] * self.n + pred[keep]
        self.mat += np.bincount(idx, minlength=self.n**2).reshape(self.n, self.n)

    def iou(self) -> np.ndarray:
        tp = np.diag(self.mat)
        denom = self.mat.sum(0) + self.mat.sum(1) - tp
        return np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)

    def miou(self) -> float:
        return float(np.nanmean(self.iou()))

    def overall_accuracy(self) -> float:
        return float(np.diag(self.mat).sum() / max(self.mat.sum(), 1))

    def per_class_recall(self) -> np.ndarray:
        rows = self.mat.sum(1)
        return np.where(rows > 0, np.diag(self.mat) / np.maximum(rows, 1), np.nan)

    def balanced_accuracy(self) -> float:
        return float(np.nanmean(self.per_class_recall()))

    def f1(self) -> np.ndarray:
        tp = np.diag(self.mat)
        denom = self.mat.sum(0) + self.mat.sum(1)
        return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), np.nan)

    def macro_f1(self) -> float:
        return float(np.nanmean(self.f1()))

    def summary(self) -> dict[str, float]:
        return {
            "oa": self.overall_accuracy(),
            "miou": self.miou(),
            "balanced_acc": self.balanced_accuracy(),
            "macro_f1": self.macro_f1(),
        }
