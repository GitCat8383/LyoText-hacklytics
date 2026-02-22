#!/usr/bin/env python3
"""Train EEGNet on a single labeled CSV (e.g. train1.csv).

Handles CSVs with columns: time_interval, TP9, AF7, AF8, TP10, target_class
where target_class can be "blinking", "jaw clenching", etc.

When the CSV has no idle class, idle epochs are mined automatically by
detecting low-amplitude windows (the subject cannot gesture continuously
over multi-minute segments).

Feature engineering pipeline:
    1. Load full recording, map labels
    2. Bandpass filter 1-100 Hz (recording-level)
    3. Common Average Re-referencing (recording-level)
    4. Sliding 1s windows with 50% overlap
    5. Amplitude-based classification:
       - idle:    all channels peak-to-peak < idle_threshold
       - gesture: relevant channels peak-to-peak > gesture_threshold
       - discard: ambiguous windows
    6. Artifact rejection (peak-to-peak > 800 uV)
    7. Per-epoch z-score normalization + band-power features -> 24 channels
    8. Class balancing via oversampling minority class
    9. Train EEGNet with confusion matrix + classification report

Usage:
    python train_single_csv.py --csv kaggle_data/train1.csv
    python train_single_csv.py --csv kaggle_data/train1.csv --idle-threshold 50 --gesture-threshold 100
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from eeg.dataset import (
    LABEL_MAP, LABEL_NAMES, save_epochs,
    bandpass_filter, common_avg_reference, reject_artifact,
    normalize_epoch, compute_band_powers,
)
from eeg.deep_trainer import DeepTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("train_single_csv")

SAMPLE_RATE = 256
WINDOW_SAMPLES = 256
EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
FRONTAL_IDX = [1, 2]   # AF7, AF8
TEMPORAL_IDX = [0, 3]  # TP9, TP10

LABEL_REMAP = {
    "blinking": "blink",
    "jaw clenching": "clench",
    "clenching": "clench",
    "blink": "blink",
    "clench": "clench",
    "idle": "idle",
}


def _preprocess_window(window: np.ndarray, fs: int = SAMPLE_RATE) -> np.ndarray:
    """Z-score normalize + append 20 band-power channels. Input already filtered."""
    normalized = normalize_epoch(window)
    band_feats = compute_band_powers(window, fs=fs)
    return np.concatenate([normalized, band_feats], axis=0).astype(np.float32)


def extract_epochs(
    eeg: np.ndarray,
    labels_per_sample: np.ndarray,
    idle_threshold: float = 50.0,
    gesture_threshold: float = 100.0,
) -> tuple[list[np.ndarray], list[str]]:
    """Extract and classify 1s windows using amplitude-based idle mining.

    Args:
        eeg: (n_samples, 4) filtered+CAR'd EEG
        labels_per_sample: original label per sample ("blink" or "clench")
        idle_threshold: max peak-to-peak (uV) across all channels for idle
        gesture_threshold: min peak-to-peak (uV) on relevant channels for gesture
    """
    epochs: list[np.ndarray] = []
    labels: list[str] = []
    stride = WINDOW_SAMPLES // 2
    discarded = 0
    rejected = 0

    for s in range(0, len(eeg) - WINDOW_SAMPLES + 1, stride):
        window = eeg[s : s + WINDOW_SAMPLES].T  # (4, 256)

        if not reject_artifact(window):
            rejected += 1
            continue

        pp = window.max(axis=1) - window.min(axis=1)  # per-channel peak-to-peak
        max_pp = float(pp.max())
        original_label = labels_per_sample[s]

        if max_pp < idle_threshold:
            epochs.append(_preprocess_window(window))
            labels.append("idle")
        elif original_label == "blink" and float(pp[FRONTAL_IDX].max()) > gesture_threshold:
            epochs.append(_preprocess_window(window))
            labels.append("blink")
        elif original_label == "clench" and float(pp[TEMPORAL_IDX].max()) > gesture_threshold:
            epochs.append(_preprocess_window(window))
            labels.append("clench")
        else:
            discarded += 1

    logger.info(
        "Extracted %d epochs (%d discarded, %d artifact-rejected)",
        len(epochs), discarded, rejected,
    )
    return epochs, labels


def _balance_classes(
    epochs: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample minority classes to match the majority class count."""
    unique, counts = np.unique(labels, return_counts=True)
    max_count = counts.max()

    bal_epochs, bal_labels = [epochs], [labels]

    for cls, cnt in zip(unique, counts):
        if cnt < max_count:
            deficit = max_count - cnt
            idx = np.where(labels == cls)[0]
            rng = np.random.default_rng(42)
            extra_idx = rng.choice(idx, size=deficit, replace=True)
            bal_epochs.append(epochs[extra_idx])
            bal_labels.append(labels[extra_idx])

    return np.concatenate(bal_epochs), np.concatenate(bal_labels)


def _print_results(title, result):
    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE — {title}")
    print(f"{'=' * 60}")
    print(f"  Best val accuracy:  {result.best_val_accuracy:.1%}")
    print(f"  Best val loss:      {result.best_val_loss:.4f}")
    print(f"  Train accuracy:     {result.final_train_accuracy:.1%}")
    print(f"  Epochs trained:     {result.epochs_trained}")
    print(f"  Total time:         {result.total_time_sec:.1f}s")
    print(f"  Model saved to:     {result.model_path}")
    print()

    targets = {"idle": 0.65, "blink": 0.80, "clench": 0.70}
    print(f"  Per-class accuracy:")
    for cls, acc in result.class_accuracies.items():
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        target = targets.get(cls, 0.65)
        status = "✓" if acc >= target else "✗"
        print(f"    {status} {cls:<10} {bar} {acc:.1%}  (target: {target:.0%})")
    print()

    if result.confusion_matrix is not None and result.class_names:
        cm = result.confusion_matrix
        names = result.class_names
        print(f"  CONFUSION MATRIX (rows=true, cols=predicted)")
        header = "         " + "".join(f"{n[:7]:>8}" for n in names)
        print(f"  {header}")
        for i, name in enumerate(names):
            row = "".join(f"{cm[i, j]:>8d}" for j in range(len(names)))
            print(f"    {name[:7]:<7} {row}")
        print()

    if result.classification_report:
        print(f"  CLASSIFICATION REPORT")
        for line in result.classification_report.split("\n"):
            if line.strip():
                print(f"    {line}")
        print()

    print(f"{'=' * 60}")
    print()
    print(f"  Model saved with: weights + label_map + preprocessing_params")
    print(f"  To load into running backend:")
    print(f"    curl -X POST http://localhost:8000/api/dl/models/reload")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Train EEGNet on a single labeled CSV with amplitude-based idle mining"
    )
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to the labeled CSV file")
    parser.add_argument("--idle-threshold", type=float, default=50.0,
                        help="Max peak-to-peak (uV) for idle classification (default: 50)")
    parser.add_argument("--gesture-threshold", type=float, default=100.0,
                        help="Min peak-to-peak (uV) for gesture classification (default: 100)")
    parser.add_argument("--epochs", type=int, default=200, help="Max training epochs")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--session-name", type=str, default="single_csv_train",
                        help="Name for saved session")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error("CSV file not found: %s", csv_path)
        sys.exit(1)

    # ── Phase 1: Load CSV ─────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Loading {csv_path.name}...")
    print(f"{'=' * 60}")

    df = pd.read_csv(str(csv_path), low_memory=False)
    logger.info("Loaded %d rows from %s", len(df), csv_path)

    missing = [c for c in EEG_CHANNELS if c not in df.columns]
    if missing:
        logger.error("Missing EEG columns: %s (have: %s)", missing, list(df.columns))
        sys.exit(1)
    if "target_class" not in df.columns:
        logger.error("Missing 'target_class' column")
        sys.exit(1)

    eeg = df[EEG_CHANNELS].values.astype(np.float32)
    valid_mask = ~np.isnan(eeg).any(axis=1)
    eeg = eeg[valid_mask]

    raw_labels = df["target_class"].values[valid_mask]
    mapped_labels = np.array([LABEL_REMAP.get(l, l) for l in raw_labels])

    raw_counts = {}
    for l in raw_labels:
        raw_counts[l] = raw_counts.get(l, 0) + 1

    print(f"  Samples:  {len(eeg):,}")
    print(f"  Duration: {len(eeg) / SAMPLE_RATE / 60:.1f} min")
    print(f"  Original classes:")
    for cls, n in sorted(raw_counts.items()):
        print(f"    {cls}: {n:,}")
    has_idle = "idle" in set(mapped_labels)
    print(f"  Has idle class: {'yes' if has_idle else 'NO — will mine from low-amplitude windows'}")
    print()

    # ── Phase 2: Recording-level filtering ────────────────────
    print(f"  Applying bandpass 1-100 Hz + CAR (recording-level)...")

    # Process in chunks to avoid memory issues with 1.95M samples
    CHUNK = 500_000
    eeg_t = eeg.T  # (4, n_samples)
    filtered = np.zeros_like(eeg_t)
    for start in range(0, eeg_t.shape[1], CHUNK):
        end = min(start + CHUNK, eeg_t.shape[1])
        chunk = eeg_t[:, start:end]
        if chunk.shape[1] > 50:
            filtered[:, start:end] = bandpass_filter(chunk, low=1.0, high=100.0, fs=SAMPLE_RATE)
        else:
            filtered[:, start:end] = chunk
    filtered = common_avg_reference(filtered)
    eeg = filtered.T  # (n_samples, 4)

    print(f"  Filtering complete.")
    print()

    # ── Phase 3: Extract epochs with idle mining ──────────────
    print(f"  Extracting epochs (idle_threshold={args.idle_threshold} uV, gesture_threshold={args.gesture_threshold} uV)...")

    all_epochs, all_labels = extract_epochs(
        eeg, mapped_labels,
        idle_threshold=args.idle_threshold,
        gesture_threshold=args.gesture_threshold,
    )

    if not all_epochs:
        logger.error("No epochs extracted. Try adjusting thresholds.")
        sys.exit(1)

    counts: dict[str, int] = {}
    for l in all_labels:
        counts[l] = counts.get(l, 0) + 1

    print(f"\n{'=' * 60}")
    print(f"  EPOCH EXTRACTION COMPLETE")
    print(f"  Pipeline: bandpass 1-100Hz -> CAR -> amplitude classify -> z-score -> band powers")
    print(f"{'=' * 60}")
    print(f"  Total epochs:  {len(all_epochs)}")
    print(f"  Epoch shape:   {all_epochs[0].shape} (24-ch: 4 EEG + 20 band-power)")
    for cls, n in sorted(counts.items()):
        pct = n / len(all_epochs) * 100
        print(f"    {cls:<10}: {n:>6}  ({pct:.1f}%)")
    print(f"{'=' * 60}\n")

    # ── Phase 4: Map to integer labels ────────────────────────
    label_to_int = {"idle": 0, "blink": 2, "clench": 3}
    mapped_int = []
    mapped_epochs = []
    for i, l in enumerate(all_labels):
        if l in label_to_int:
            mapped_int.append(label_to_int[l])
            mapped_epochs.append(all_epochs[i])

    epochs_arr = np.array(mapped_epochs, dtype=np.float32)
    labels_arr = np.array(mapped_int, dtype=np.int64)

    # ── Phase 5: Balance classes ──────────────────────────────
    unique_cls, cls_counts = np.unique(labels_arr, return_counts=True)
    imbalance_ratio = cls_counts.max() / max(cls_counts.min(), 1)

    if imbalance_ratio > 1.5:
        print(f"  Class imbalance detected (ratio {imbalance_ratio:.1f}x). Oversampling minority...")
        epochs_arr, labels_arr = _balance_classes(epochs_arr, labels_arr)
        unique_cls2, cls_counts2 = np.unique(labels_arr, return_counts=True)
        for c, n in zip(unique_cls2, cls_counts2):
            name = LABEL_NAMES[c] if c < len(LABEL_NAMES) else str(c)
            print(f"    {name:<10}: {n:>6}")
        print(f"  Balanced total: {len(labels_arr)}")
        print()

    # ── Phase 6: Save preprocessed data ───────────────────────
    save_path = save_epochs(
        epochs=[e for e in epochs_arr],
        labels=[LABEL_NAMES[l] if l < len(LABEL_NAMES) else str(l) for l in labels_arr],
        session_name=args.session_name,
        metadata={
            "source": "single_csv",
            "csv_file": str(csv_path),
            "idle_threshold": args.idle_threshold,
            "gesture_threshold": args.gesture_threshold,
            "preprocessing": "bandpass_1-100Hz + CAR + z-score + band_powers (24-ch)",
            "balanced": bool(imbalance_ratio > 1.5),
        },
    )
    logger.info("Saved preprocessed data to %s", save_path)

    # ── Phase 7: Train ────────────────────────────────────────
    logger.info(
        "Training EEGNet on %d epochs (%d classes)...",
        len(labels_arr), len(set(labels_arr.tolist())),
    )

    trainer = DeepTrainer()
    result = trainer.train_gesture(
        epochs_arr, labels_arr,
        max_epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
    )

    _print_results(f"train1.csv ({csv_path.name})", result)


if __name__ == "__main__":
    main()
