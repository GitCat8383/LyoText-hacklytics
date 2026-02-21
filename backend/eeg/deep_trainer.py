"""Training pipeline for EEGNet models.

Handles training, validation, early stopping, model persistence,
and real-time progress reporting via the event bus.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import config
from eeg.eegnet import EEGNet, create_p300_model, create_gesture_model, create_unified_model
from eeg.dataset import (
    EEGDataset, create_dataloaders, load_all_sessions,
    load_epochs, save_epochs, LABEL_NAMES,
)
from utils.events import Event, EventType, event_bus

logger = logging.getLogger(__name__)

MODELS_DIR = Path(config.BASE_DIR) / "models"


@dataclass
class TrainResult:
    best_val_accuracy: float
    best_val_loss: float
    final_train_accuracy: float
    epochs_trained: int
    total_time_sec: float
    model_path: str
    class_accuracies: dict[str, float]


class DeepTrainer:
    """Manages training lifecycle for EEGNet models.

    Inference is optimized for minimal latency:
    - Models stay in memory permanently after loading
    - Pre-allocated input tensors avoid repeated allocation
    - TorchScript tracing eliminates Python overhead
    - Warmup forward passes prime CPU caches
    """

    def __init__(self) -> None:
        self._device = torch.device("cpu")
        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
        logger.info("DeepTrainer using device: %s", self._device)

        self._p300_model: EEGNet | None = None
        self._gesture_model: EEGNet | None = None
        self._training = False

        # Pre-allocated tensors for zero-alloc inference
        self._gesture_input: torch.Tensor | None = None
        self._p300_input: torch.Tensor | None = None

        # TorchScript traced models for faster inference
        self._gesture_traced: torch.jit.ScriptModule | None = None
        self._p300_traced: torch.jit.ScriptModule | None = None

    @property
    def is_training(self) -> bool:
        return self._training

    @property
    def p300_model(self) -> EEGNet | None:
        return self._p300_model

    @property
    def gesture_model(self) -> EEGNet | None:
        return self._gesture_model

    def load_models(self) -> dict[str, bool]:
        """Load saved models from disk, trace them, and warm up for instant inference."""
        results = {}

        p300_path = MODELS_DIR / "eegnet_p300.pt"
        if p300_path.exists():
            try:
                state = torch.load(p300_path, map_location=self._device, weights_only=True)
                self._p300_model = create_p300_model(n_samples=state["n_samples"])
                self._p300_model.load_state_dict(state["model_state"])
                self._p300_model.to(self._device).eval()

                self._p300_input = torch.zeros(
                    1, 1, 4, state["n_samples"],
                    dtype=torch.float32, device=self._device,
                )
                self._trace_and_warmup("p300", self._p300_model, self._p300_input)
                results["p300"] = True
                logger.info("Loaded P300 EEGNet (%d params, traced + warmed up)",
                            self._p300_model.count_parameters())
            except Exception:
                logger.exception("Failed to load P300 model")
                results["p300"] = False
        else:
            results["p300"] = False

        gesture_path = MODELS_DIR / "eegnet_gesture.pt"
        if gesture_path.exists():
            try:
                state = torch.load(gesture_path, map_location=self._device, weights_only=True)
                # Detect actual n_classes from the saved classifier weights
                n_classes = state["model_state"]["classifier.weight"].shape[0]
                self._gesture_model = create_gesture_model(n_samples=state["n_samples"])
                if n_classes != self._gesture_model.n_classes:
                    self._gesture_model.classifier = nn.Linear(
                        self._gesture_model._flat_size, n_classes
                    )
                    self._gesture_model.n_classes = n_classes
                self._gesture_model.load_state_dict(state["model_state"])
                self._gesture_model.to(self._device).eval()

                self._gesture_input = torch.zeros(
                    1, 1, 4, state["n_samples"],
                    dtype=torch.float32, device=self._device,
                )
                self._trace_and_warmup("gesture", self._gesture_model, self._gesture_input)
                results["gesture"] = True
                logger.info("Loaded gesture EEGNet (%d params, traced + warmed up)",
                            self._gesture_model.count_parameters())
            except Exception:
                logger.exception("Failed to load gesture model")
                results["gesture"] = False
        else:
            results["gesture"] = False

        return results

    def _trace_and_warmup(self, name: str, model: EEGNet, dummy_input: torch.Tensor) -> None:
        """Trace the model with TorchScript and run warmup passes."""
        try:
            traced = torch.jit.trace(model, dummy_input)
            traced = torch.jit.freeze(traced)
            if name == "p300":
                self._p300_traced = traced
            else:
                self._gesture_traced = traced

            # Warmup: 5 forward passes to prime CPU caches and JIT
            for _ in range(5):
                with torch.no_grad():
                    traced(dummy_input)
            logger.info("[%s] TorchScript traced + 5 warmup passes done", name)
        except Exception:
            logger.warning("[%s] TorchScript tracing failed, using eager mode", name)
            if name == "p300":
                self._p300_traced = None
            else:
                self._gesture_traced = None

    def train_p300(
        self,
        epochs_data: np.ndarray,
        labels: np.ndarray,
        max_epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        patience: int = 15,
    ) -> TrainResult:
        """Train P300 model (binary: target=1 vs non-target=0)."""
        n_samples = epochs_data.shape[2]
        model = create_p300_model(n_samples=n_samples)
        result = self._train(
            model=model,
            epochs_data=epochs_data,
            labels=labels,
            model_name="p300",
            max_epochs=max_epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            class_names=["non_target", "target"],
        )
        self._p300_model = model
        return result

    def train_gesture(
        self,
        epochs_data: np.ndarray,
        labels: np.ndarray,
        max_epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        patience: int = 15,
    ) -> TrainResult:
        """Train gesture model (idle=0, blink=2, clench=3, noise=4 → remapped to 0-3)."""
        # Remap labels to contiguous 0..3
        unique_labels = sorted(set(labels.tolist()))
        label_remap = {old: new for new, old in enumerate(unique_labels)}
        remapped = np.array([label_remap[l] for l in labels])
        class_names = [LABEL_NAMES[l] if l < len(LABEL_NAMES) else str(l) for l in unique_labels]

        n_samples = epochs_data.shape[2]
        model = create_gesture_model(n_samples=n_samples)
        model.classifier = nn.Linear(model._flat_size, len(unique_labels))
        model.n_classes = len(unique_labels)

        result = self._train(
            model=model,
            epochs_data=epochs_data,
            labels=remapped,
            model_name="gesture",
            max_epochs=max_epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            class_names=class_names,
        )
        self._gesture_model = model
        return result

    def _train(
        self,
        model: EEGNet,
        epochs_data: np.ndarray,
        labels: np.ndarray,
        model_name: str,
        max_epochs: int,
        batch_size: int,
        lr: float,
        patience: int,
        class_names: list[str],
    ) -> TrainResult:
        self._training = True
        start_time = time.time()

        model = model.to(self._device)
        logger.info(
            "Training %s EEGNet: %d epochs, %d classes, %d params",
            model_name, len(labels), len(class_names), model.count_parameters(),
        )

        # Class-weighted loss for imbalanced data
        class_counts = np.bincount(labels, minlength=len(class_names)).astype(np.float32)
        class_counts = np.maximum(class_counts, 1.0)
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * len(class_names)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32).to(self._device)
        )

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

        train_loader, val_loader = create_dataloaders(
            epochs_data, labels, batch_size=batch_size, augment_train=True,
        )

        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(max_epochs):
            # ── Train ──
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            for X, y in train_loader:
                X, y = X.to(self._device), y.to(self._device)
                optimizer.zero_grad()
                logits = model(X)
                loss = criterion(logits, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item() * len(y)
                train_correct += (logits.argmax(1) == y).sum().item()
                train_total += len(y)

            train_loss /= train_total
            train_acc = train_correct / train_total

            # ── Validate ──
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            all_preds, all_true = [], []
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(self._device), y.to(self._device)
                    logits = model(X)
                    loss = criterion(logits, y)
                    val_loss += loss.item() * len(y)
                    preds = logits.argmax(1)
                    val_correct += (preds == y).sum().item()
                    val_total += len(y)
                    all_preds.extend(preds.cpu().numpy())
                    all_true.extend(y.cpu().numpy())

            val_loss /= max(val_total, 1)
            val_acc = val_correct / max(val_total, 1)
            scheduler.step(val_loss)

            # ── Progress reporting ──
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    "[%s] Epoch %d/%d — train_loss=%.4f train_acc=%.1f%% val_loss=%.4f val_acc=%.1f%%",
                    model_name, epoch + 1, max_epochs,
                    train_loss, train_acc * 100, val_loss, val_acc * 100,
                )

            event_bus.emit(Event(
                type=EventType.CALIBRATION_PROGRESS,
                data={
                    "model": model_name,
                    "epoch": epoch + 1,
                    "max_epochs": max_epochs,
                    "train_loss": round(train_loss, 4),
                    "train_acc": round(train_acc, 4),
                    "val_loss": round(val_loss, 4),
                    "val_acc": round(val_acc, 4),
                },
            ))

            # ── Early stopping ──
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info("[%s] Early stopping at epoch %d", model_name, epoch + 1)
                    break

        # Restore best model
        if best_state:
            model.load_state_dict(best_state)
        model.to(self._device).eval()

        # Per-class accuracy
        all_preds_np = np.array(all_preds)
        all_true_np = np.array(all_true)
        class_accs = {}
        for i, name in enumerate(class_names):
            mask = all_true_np == i
            if mask.sum() > 0:
                class_accs[name] = float((all_preds_np[mask] == i).mean())

        # Save model
        save_path = self._save_model(model, model_name, epochs_data.shape[2])
        total_time = time.time() - start_time

        self._training = False

        result = TrainResult(
            best_val_accuracy=best_val_acc,
            best_val_loss=best_val_loss,
            final_train_accuracy=train_acc,
            epochs_trained=epoch + 1,
            total_time_sec=total_time,
            model_path=str(save_path),
            class_accuracies=class_accs,
        )
        logger.info(
            "[%s] Training complete: val_acc=%.1f%% in %.1fs (%d epochs)",
            model_name, best_val_acc * 100, total_time, epoch + 1,
        )
        return result

    def _save_model(self, model: EEGNet, name: str, n_samples: int) -> Path:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = MODELS_DIR / f"eegnet_{name}.pt"
        torch.save({
            "model_state": model.state_dict(),
            "n_channels": model.n_channels,
            "n_samples": n_samples,
            "n_classes": model.n_classes,
        }, path)
        logger.info("Saved %s model to %s", name, path)

        # Pre-allocate input tensor and trace for instant inference
        dummy = torch.zeros(1, 1, 4, n_samples, dtype=torch.float32, device=self._device)
        if name == "p300":
            self._p300_input = dummy
        else:
            self._gesture_input = dummy
        self._trace_and_warmup(name, model, dummy)

        return path

    # ── Real-time inference (zero-alloc, traced) ───────────────

    def predict_p300(self, epoch: np.ndarray) -> tuple[int, float]:
        """Predict P300 target/non-target for a single epoch.

        Uses pre-allocated tensor + TorchScript traced model for minimum latency.
        No new memory allocation per call.
        """
        if self._p300_model is None or self._p300_input is None:
            raise RuntimeError("P300 model not loaded")

        # Copy data into pre-allocated tensor (no new allocation)
        self._p300_input[0, 0].copy_(
            torch.from_numpy(epoch.astype(np.float32))
        )

        model = self._p300_traced or self._p300_model
        with torch.no_grad():
            logits = model(self._p300_input)
            probs = torch.softmax(logits, dim=1)
            pred = probs[0].argmax().item()
            confidence = probs[0, pred].item()

        return pred, confidence

    def predict_gesture(self, window: np.ndarray) -> tuple[int, str, float]:
        """Classify a gesture from a raw EEG window.

        Uses pre-allocated tensor + TorchScript traced model for minimum latency.
        Applies per-channel z-score normalization to match training distribution.
        """
        if self._gesture_model is None or self._gesture_input is None:
            raise RuntimeError("Gesture model not loaded")

        from eeg.dataset import normalize_epoch
        normalized = normalize_epoch(window.astype(np.float32))

        self._gesture_input[0, 0].copy_(
            torch.from_numpy(normalized)
        )

        model = self._gesture_traced or self._gesture_model
        with torch.no_grad():
            logits = model(self._gesture_input)
            probs = torch.softmax(logits, dim=1)
            pred = probs[0].argmax().item()
            confidence = probs[0, pred].item()

        gesture_names = ["idle", "blink", "clench", "noise"]
        name = gesture_names[pred] if pred < len(gesture_names) else f"class_{pred}"
        return pred, name, confidence

    def select_phrase_deep(
        self,
        epochs: list[tuple[np.ndarray, int]],
        num_phrases: int = config.NUM_PHRASES,
    ) -> tuple[int, float]:
        """P300-based phrase selection using EEGNet (drop-in replacement for LDA)."""
        if self._p300_model is None:
            raise RuntimeError("P300 model not loaded")

        scores = np.zeros(num_phrases)
        counts = np.zeros(num_phrases)

        for epoch, phrase_idx in epochs:
            if 0 <= phrase_idx < num_phrases:
                pred, confidence = self.predict_p300(epoch)
                if pred == 1:  # target
                    scores[phrase_idx] += confidence
                counts[phrase_idx] += 1

        with np.errstate(divide="ignore", invalid="ignore"):
            avg_scores = np.where(counts > 0, scores / counts, 0)

        winner = int(np.argmax(avg_scores))
        winning_confidence = float(avg_scores[winner])
        return winner, winning_confidence


deep_trainer = DeepTrainer()
