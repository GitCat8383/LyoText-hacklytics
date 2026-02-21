"""EEG dataset utilities: loading, saving, augmentation, and torch DataLoaders."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import config

logger = logging.getLogger(__name__)

LABEL_MAP = {
    "idle": 0,
    "p300_target": 1,
    "blink": 2,
    "clench": 3,
    "noise": 4,
    # Binary P300 aliases
    "non_target": 0,
    "target": 1,
}

LABEL_NAMES = ["idle", "p300_target", "blink", "clench", "noise"]

DATA_DIR = Path(config.BASE_DIR) / "data"


def normalize_epoch(epoch: np.ndarray) -> np.ndarray:
    """Per-channel z-score normalization for a single epoch.

    Args:
        epoch: shape (n_channels, n_samples)
    Returns:
        Normalized epoch with each channel having mean≈0, std≈1.
        Channels with zero variance are left as zeros.
    """
    mean = epoch.mean(axis=1, keepdims=True)
    std = epoch.std(axis=1, keepdims=True)
    std[std < 1e-8] = 1.0
    return (epoch - mean) / std


class EEGDataset(Dataset):
    """In-memory EEG epoch dataset with optional augmentation."""

    def __init__(
        self,
        epochs: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
    ) -> None:
        """
        Args:
            epochs: shape (n_epochs, n_channels, n_samples)
            labels: shape (n_epochs,) integer class labels
            augment: apply real-time data augmentation
        """
        self.epochs = epochs.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        epoch = self.epochs[idx].copy()

        if self.augment:
            epoch = self._augment(epoch)

        epoch = normalize_epoch(epoch)

        # Shape: (1, n_channels, n_samples) — add the "image channel" dim
        x = torch.tensor(epoch, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

    def _augment(self, epoch: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng()

        # Gaussian noise injection (σ = 5–15% of signal std)
        if rng.random() < 0.5:
            noise_scale = rng.uniform(0.05, 0.15) * epoch.std()
            epoch = epoch + rng.normal(0, noise_scale, epoch.shape)

        # Amplitude scaling (0.8×–1.2×)
        if rng.random() < 0.5:
            scale = rng.uniform(0.8, 1.2)
            epoch = epoch * scale

        # Temporal shift (±10 samples)
        if rng.random() < 0.3:
            shift = rng.integers(-10, 11)
            epoch = np.roll(epoch, shift, axis=1)

        # Channel dropout (zero one random channel)
        if rng.random() < 0.2:
            ch = rng.integers(0, epoch.shape[0])
            epoch[ch, :] = 0.0

        return epoch


# ── Saving / loading collected data ───────────────────────────

def save_epochs(
    epochs: list[np.ndarray],
    labels: list[str],
    session_name: str,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save collected epochs to disk for later training.

    Args:
        epochs: list of arrays, each shape (n_channels, n_samples)
        labels: list of string labels (e.g. "blink", "idle", "target")
        session_name: unique name for this collection session
        metadata: optional extra info (subject, date, notes)

    Returns:
        Path to the saved .npz file.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_path = DATA_DIR / f"{session_name}.npz"

    epochs_arr = np.array(epochs, dtype=np.float32)
    labels_int = np.array([LABEL_MAP.get(l, 0) for l in labels], dtype=np.int64)

    np.savez_compressed(
        save_path,
        epochs=epochs_arr,
        labels=labels_int,
        label_names=np.array(labels),
    )

    meta_path = DATA_DIR / f"{session_name}_meta.json"
    meta = metadata or {}
    meta.update({
        "n_epochs": len(epochs),
        "n_channels": epochs_arr.shape[1] if epochs_arr.ndim == 3 else 0,
        "n_samples": epochs_arr.shape[2] if epochs_arr.ndim == 3 else 0,
        "class_distribution": {
            name: int((labels_int == idx).sum())
            for idx, name in enumerate(LABEL_NAMES)
            if (labels_int == idx).sum() > 0
        },
    })
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info("Saved %d epochs to %s", len(epochs), save_path)
    return save_path


def load_epochs(session_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load epochs from a saved session.

    Returns:
        (epochs, labels) where epochs shape is (n, channels, samples)
        and labels shape is (n,) with integer class labels.
    """
    path = DATA_DIR / f"{session_name}.npz"
    data = np.load(path)
    return data["epochs"], data["labels"]


def load_all_sessions() -> tuple[np.ndarray, np.ndarray]:
    """Load and concatenate all saved sessions.

    Returns:
        (epochs, labels) concatenated from all .npz files in DATA_DIR.
    """
    if not DATA_DIR.exists():
        return np.array([]), np.array([])

    all_epochs, all_labels = [], []
    for npz_file in sorted(DATA_DIR.glob("*.npz")):
        data = np.load(npz_file)
        all_epochs.append(data["epochs"])
        all_labels.append(data["labels"])
        logger.info("Loaded %d epochs from %s", len(data["labels"]), npz_file.name)

    if not all_epochs:
        return np.array([]), np.array([])

    return np.concatenate(all_epochs), np.concatenate(all_labels)


def list_sessions() -> list[dict[str, Any]]:
    """List all saved data sessions with metadata."""
    if not DATA_DIR.exists():
        return []

    sessions = []
    for npz_file in sorted(DATA_DIR.glob("*.npz")):
        name = npz_file.stem
        meta_path = DATA_DIR / f"{name}_meta.json"
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())

        data = np.load(npz_file)
        sessions.append({
            "name": name,
            "n_epochs": len(data["labels"]),
            "file_size_kb": npz_file.stat().st_size // 1024,
            **meta,
        })

    return sessions


def create_dataloaders(
    epochs: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
    val_split: float = 0.2,
    augment_train: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Split data and create train/val DataLoaders with stratified split."""
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
    train_idx, val_idx = next(splitter.split(epochs, labels))

    train_ds = EEGDataset(epochs[train_idx], labels[train_idx], augment=augment_train)
    val_ds = EEGDataset(epochs[val_idx], labels[val_idx], augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
