"""Metricas binarias (SLICK = positivo), acumuladas a nivel de pixel sobre todo el split.

Mismo criterio que los baselines del dataset (matriz de confusion global, micro-promediada),
asi que iou_slick es directamente comparable con BASELINE_ML.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-9


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        pred = np.asarray(pred).astype(bool)
        target = np.asarray(target).astype(bool)
        self.tp += int(np.logical_and(pred, target).sum())
        self.fp += int(np.logical_and(pred, ~target).sum())
        self.fn += int(np.logical_and(~pred, target).sum())
        self.tn += int(np.logical_and(~pred, ~target).sum())

    def scores(self) -> dict:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        return {
            "iou": tp / (tp + fp + fn + _EPS),
            "dice": 2 * tp / (2 * tp + fp + fn + _EPS),
            "precision": tp / (tp + fp + _EPS),
            "recall": tp / (tp + fn + _EPS),
            "pixel_accuracy": (tp + tn) / (tp + fp + fn + tn + _EPS),
            "pixels": tp + fp + fn + tn,
        }
