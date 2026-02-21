"""P300 classifier using Linear Discriminant Analysis.

Trained once during calibration, persisted to disk with joblib,
and loaded on subsequent startups.
"""

from __future__ import annotations

import logging
import os

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import config

logger = logging.getLogger(__name__)


def _extract_features(epoch: np.ndarray) -> np.ndarray:
    """Downsample and flatten an epoch into a feature vector."""
    ds = config.DOWNSAMPLE_FACTOR
    downsampled = epoch[:, ::ds]
    return downsampled.flatten()


class P300Classifier:

    def __init__(self) -> None:
        self._model: LinearDiscriminantAnalysis | None = None
        self._is_trained = False
        self._cal_epochs: list[np.ndarray] = []
        self._cal_labels: list[int] = []
        self._calibrating = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def is_calibrating(self) -> bool:
        return self._calibrating

    @property
    def calibration_count(self) -> int:
        return len(self._cal_epochs)

    def load(self, path: str | None = None) -> bool:
        path = path or config.MODEL_PATH
        if not os.path.exists(path):
            logger.info("No saved model at %s", path)
            return False
        try:
            self._model = joblib.load(path)
            self._is_trained = True
            logger.info("Loaded P300 model from %s", path)
            return True
        except Exception:
            logger.exception("Failed to load model from %s", path)
            return False

    def save(self, path: str | None = None) -> None:
        path = path or config.MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self._model, path)
        logger.info("Saved P300 model to %s", path)

    def start_calibration(self) -> None:
        self._cal_epochs.clear()
        self._cal_labels.clear()
        self._calibrating = True
        logger.info("Calibration started")

    def add_calibration_epoch(self, epoch: np.ndarray, is_target: bool) -> int:
        self._cal_epochs.append(epoch)
        self._cal_labels.append(1 if is_target else 0)
        return len(self._cal_epochs)

    def finish_calibration(self) -> float:
        self._calibrating = False

        if len(self._cal_epochs) < 10:
            raise ValueError(
                f"Need at least 10 epochs for training, got {len(self._cal_epochs)}"
            )

        X = np.array([_extract_features(e) for e in self._cal_epochs])
        y = np.array(self._cal_labels)

        self._model = LinearDiscriminantAnalysis()
        self._model.fit(X, y)
        self._is_trained = True

        accuracy = self._model.score(X, y)
        logger.info(
            "Calibration complete: %d epochs, training accuracy=%.2f%%",
            len(self._cal_epochs),
            accuracy * 100,
        )

        self.save()
        self._cal_epochs.clear()
        self._cal_labels.clear()
        return accuracy

    def predict_epoch(self, epoch: np.ndarray) -> tuple[int, float]:
        if not self._is_trained or self._model is None:
            raise RuntimeError("Classifier not trained")

        features = _extract_features(epoch).reshape(1, -1)
        pred = self._model.predict(features)[0]
        proba = self._model.predict_proba(features)[0]
        target_confidence = proba[1] if len(proba) > 1 else proba[0]
        return int(pred), float(target_confidence)

    def select_phrase(
        self,
        epochs: list[tuple[np.ndarray, int]],
        num_phrases: int = config.NUM_PHRASES,
    ) -> tuple[int, float]:
        if not self._is_trained or self._model is None:
            raise RuntimeError("Classifier not trained")

        scores = np.zeros(num_phrases)
        counts = np.zeros(num_phrases)

        for epoch, phrase_idx in epochs:
            if 0 <= phrase_idx < num_phrases:
                _, confidence = self.predict_epoch(epoch)
                scores[phrase_idx] += confidence
                counts[phrase_idx] += 1

        with np.errstate(divide="ignore", invalid="ignore"):
            avg_scores = np.where(counts > 0, scores / counts, 0)

        winner = int(np.argmax(avg_scores))
        winning_confidence = float(avg_scores[winner])
        return winner, winning_confidence


p300_classifier = P300Classifier()
