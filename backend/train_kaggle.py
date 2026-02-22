#!/usr/bin/env python3
"""Train EEGNet on the Kaggle Muse 2 facial movements dataset.

This script handles the dataset from:
    https://www.kaggle.com/datasets/muhammadatefelkaffas/eeg-muse2-motor-imagery-brain-electrical-activity

Dataset structure:
    kaggle_data/
    └── raw_data/{subject}/{gesture}/{trial}.csv   (timestamps, TP9, AF7, AF8, TP10, Right AUX)

Gesture folders: both, left, right, eyebrows, teeth
Mapping to our labels: teeth → clench, eyebrows → blink, baseline → idle

Feature engineering pipeline (per trial):
    1. NaN removal
    2. Bandpass filter 1–100 Hz (trial-level for clean edges)
    3. Common Average Re-referencing (trial-level)
    4. Sliding-window epoch extraction (1s, 50% overlap)
    5. Artifact rejection (peak-to-peak > 800 μV)
    6. Per-epoch z-score normalization
    7. Band-power features (delta/theta/alpha/beta/gamma × 4 channels = 20 channels)
    8. Class balancing via oversampling minority class

Usage:
    python train_kaggle.py                           # preprocess + train
    python train_kaggle.py --data-dir ./kaggle_data  # custom path
    python train_kaggle.py --all-gestures            # keep all 5 gesture classes + idle
    python train_kaggle.py --loso                    # leave-one-subject-out cross-validation
    python train_kaggle.py --convert-only            # just save epochs, don't train
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
    save_epochs, LABEL_MAP, LABEL_NAMES,
    bandpass_filter, common_avg_reference, reject_artifact,
    normalize_epoch, compute_band_powers,
    create_dataloaders, EEGDataset,
)
from eeg.deep_trainer import DeepTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("train_kaggle")

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "kaggle_data")
SAMPLE_RATE = 256
WINDOW_SAMPLES = 256  # 1 second

EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]

GESTURE_TO_LABEL = {
    "teeth": "clench",
    "eyebrows": "blink",
}

ALL_GESTURE_LABELS = {
    "teeth": "clench",
    "eyebrows": "blink",
    "both": "both",
    "left": "left",
    "right": "right",
}


def _preprocess_window(window: np.ndarray, fs: int = SAMPLE_RATE) -> np.ndarray:
    """Normalize a single (4, n_samples) window and append 20 band-power channels.

    Assumes the input is already bandpass-filtered and CAR'd at the trial level.
    Returns (24, n_samples).
    """
    normalized = normalize_epoch(window)
    band_feats = compute_band_powers(window, fs=fs)
    return np.concatenate([normalized, band_feats], axis=0).astype(np.float32)


def parse_trial(
    raw_csv: str,
    gesture: str,
    label_map: dict[str, str],
) -> tuple[list[np.ndarray], list[str]]:
    """Extract preprocessed 24-channel epochs from a single trial CSV.

    Pipeline: NaN removal → trial-level bandpass (1–100 Hz) → trial-level CAR
              → sliding window → artifact rejection → z-score + band powers.
    """
    try:
        df = pd.read_csv(raw_csv, low_memory=False)
    except Exception as e:
        logger.warning("Failed to read %s: %s", raw_csv, e)
        return [], []

    if not all(c in df.columns for c in EEG_CHANNELS):
        logger.debug("Skipping %s — missing EEG columns", raw_csv)
        return [], []

    eeg = df[EEG_CHANNELS].values.astype(np.float32)
    valid_mask = ~np.isnan(eeg).any(axis=1)
    eeg = eeg[valid_mask]

    if len(eeg) < WINDOW_SAMPLES * 3:
        logger.debug("Skipping %s — too few samples (%d)", raw_csv, len(eeg))
        return [], []

    # Trial-level filtering — avoids edge artifacts from short windows
    eeg_t = eeg.T  # (4, n_samples)
    eeg_t = bandpass_filter(eeg_t, low=1.0, high=100.0, fs=SAMPLE_RATE)
    eeg_t = common_avg_reference(eeg_t)
    eeg = eeg_t.T  # (n_samples, 4)

    label = label_map.get(gesture)
    if label is None:
        return [], []

    epochs: list[np.ndarray] = []
    labels: list[str] = []
    rejected = 0
    stride = WINDOW_SAMPLES // 2  # 50% overlap

    # ── Idle: first 2 s (pre-gesture baseline) ────────────────
    idle_end = min(SAMPLE_RATE * 2, len(eeg))
    for s in range(0, idle_end - WINDOW_SAMPLES + 1, stride):
        window = eeg[s : s + WINDOW_SAMPLES].T
        if reject_artifact(window):
            epochs.append(_preprocess_window(window))
            labels.append("idle")

    # ── Idle: last 1 s (post-gesture cooldown) ────────────────
    post_start = max(len(eeg) - SAMPLE_RATE, idle_end)
    for s in range(post_start, len(eeg) - WINDOW_SAMPLES + 1, stride):
        window = eeg[s : s + WINDOW_SAMPLES].T
        if reject_artifact(window):
            epochs.append(_preprocess_window(window))
            labels.append("idle")

    # ── Gesture: middle portion (skip first 2 s, skip last 1 s) ─
    gesture_start = SAMPLE_RATE * 2
    gesture_end = len(eeg) - SAMPLE_RATE

    for s in range(gesture_start, gesture_end - WINDOW_SAMPLES + 1, stride):
        window = eeg[s : s + WINDOW_SAMPLES].T
        if reject_artifact(window):
            epochs.append(_preprocess_window(window))
            labels.append(label)
        else:
            rejected += 1

    if rejected > 0:
        logger.debug("Rejected %d artifacts in %s", rejected, raw_csv)

    return epochs, labels


def _balance_classes(
    epochs: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Oversample minority classes to match the majority class count."""
    unique, counts = np.unique(labels, return_counts=True)
    max_count = counts.max()

    bal_epochs, bal_labels, bal_subjects = [epochs], [labels], [subjects]

    for cls, cnt in zip(unique, counts):
        if cnt < max_count:
            deficit = max_count - cnt
            mask = labels == cls
            idx = np.where(mask)[0]
            rng = np.random.default_rng(42)
            extra_idx = rng.choice(idx, size=deficit, replace=True)
            bal_epochs.append(epochs[extra_idx])
            bal_labels.append(labels[extra_idx])
            bal_subjects.append(subjects[extra_idx])

    return (
        np.concatenate(bal_epochs),
        np.concatenate(bal_labels),
        np.concatenate(bal_subjects),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train EEGNet on Kaggle Muse 2 facial movements dataset"
    )
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Directory containing raw_data/")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Max training epochs")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--session-name", type=str, default="kaggle_muse2_facial",
                        help="Name for the saved training session")
    parser.add_argument("--convert-only", action="store_true",
                        help="Only convert data, don't train")
    parser.add_argument("--all-gestures", action="store_true",
                        help="Keep all 5 gesture classes instead of mapping to blink/clench")
    parser.add_argument("--loso", action="store_true",
                        help="Use leave-one-subject-out cross-validation")
    args = parser.parse_args()

    raw_dir = Path(args.data_dir) / "raw_data"

    if not raw_dir.exists():
        logger.error("raw_data directory not found at %s", raw_dir)
        logger.error("Download from: https://www.kaggle.com/datasets/muhammadatefelkaffas/eeg-muse2-motor-imagery-brain-electrical-activity")
        logger.error("Unzip to: %s", args.data_dir)
        sys.exit(1)

    label_map = ALL_GESTURE_LABELS if args.all_gestures else GESTURE_TO_LABEL

    # ── Phase 1: Parse & preprocess all trials ────────────────
    all_epochs: list[np.ndarray] = []
    all_labels: list[str] = []
    all_subjects: list[int] = []
    stats = {"trials": 0, "skipped_gesture": 0}

    subjects = sorted([d.name for d in raw_dir.iterdir() if d.is_dir()])
    logger.info("Found %d subjects: %s", len(subjects), subjects)

    for subj_idx, subject in enumerate(subjects):
        subject_raw = raw_dir / subject
        gestures = sorted([d.name for d in subject_raw.iterdir() if d.is_dir()])

        for gesture in gestures:
            if gesture not in label_map:
                stats["skipped_gesture"] += 1
                continue

            gesture_raw = subject_raw / gesture
            csv_files = sorted(gesture_raw.glob("*.csv"))
            for csv_file in csv_files:
                epochs, labels = parse_trial(str(csv_file), gesture, label_map)
                all_epochs.extend(epochs)
                all_labels.extend(labels)
                all_subjects.extend([subj_idx] * len(epochs))
                stats["trials"] += 1

    if not all_epochs:
        logger.error("No epochs extracted. Check the dataset directory structure.")
        sys.exit(1)

    # ── Phase 2: Report class distribution ────────────────────
    counts: dict[str, int] = {}
    for lbl in all_labels:
        counts[lbl] = counts.get(lbl, 0) + 1

    print(f"\n{'=' * 60}")
    print(f"  PREPROCESSING COMPLETE")
    print(f"  Pipeline: bandpass 1-100Hz → CAR → z-score → band powers")
    print(f"{'=' * 60}")
    print(f"  Subjects:         {len(subjects)}")
    print(f"  Trials processed: {stats['trials']}")
    print(f"  Total epochs:     {len(all_epochs)}")
    print(f"  Epoch shape:      {all_epochs[0].shape} (24-ch: 4 EEG + 20 band-power)")
    for cls, n in sorted(counts.items()):
        pct = n / len(all_epochs) * 100
        print(f"    {cls:<10}: {n:>5}  ({pct:.1f}%)")

    per_subj: dict[int, int] = {}
    for s in all_subjects:
        per_subj[s] = per_subj.get(s, 0) + 1
    for s, n in sorted(per_subj.items()):
        print(f"    Subject {subjects[s]}: {n} epochs")
    print(f"{'=' * 60}\n")

    # ── Phase 3: Map labels to integers ───────────────────────
    if args.all_gestures:
        extended_label_map = {**LABEL_MAP, "both": 5, "left": 6, "right": 7}
    else:
        extended_label_map = LABEL_MAP

    mapped_epochs = []
    mapped_labels = []
    mapped_subjects = []
    for i, lbl in enumerate(all_labels):
        if lbl not in extended_label_map:
            logger.warning("Unknown label '%s' at index %d, skipping", lbl, i)
            continue
        mapped_labels.append(extended_label_map[lbl])
        mapped_epochs.append(all_epochs[i])
        mapped_subjects.append(all_subjects[i])

    epochs_arr = np.array(mapped_epochs, dtype=np.float32)
    labels_arr = np.array(mapped_labels, dtype=np.int64)
    subjects_arr = np.array(mapped_subjects, dtype=np.int64)

    # ── Phase 4: Balance classes ──────────────────────────────
    unique_cls, cls_counts = np.unique(labels_arr, return_counts=True)
    imbalance_ratio = cls_counts.max() / max(cls_counts.min(), 1)

    if imbalance_ratio > 2.0:
        print(f"  Class imbalance detected (ratio {imbalance_ratio:.1f}x). Oversampling minority...")
        epochs_arr, labels_arr, subjects_arr = _balance_classes(
            epochs_arr, labels_arr, subjects_arr,
        )
        unique_cls2, cls_counts2 = np.unique(labels_arr, return_counts=True)
        for c, n in zip(unique_cls2, cls_counts2):
            name = LABEL_NAMES[c] if c < len(LABEL_NAMES) else str(c)
            print(f"    {name:<10}: {n:>5}")
        print(f"  Balanced total:   {len(labels_arr)}")
        print()

    # ── Phase 5: Save preprocessed epochs ─────────────────────
    save_path = save_epochs(
        epochs=[e for e in epochs_arr],
        labels=[LABEL_NAMES[l] if l < len(LABEL_NAMES) else str(l) for l in labels_arr],
        session_name=args.session_name,
        metadata={
            "source": "kaggle",
            "dataset": "muhammadatefelkaffas/eeg-muse2-motor-imagery-brain-electrical-activity",
            "all_gestures": bool(args.all_gestures),
            "preprocessing": "bandpass_1-100Hz + CAR + z-score + band_powers (24-ch)",
            "balanced": bool(imbalance_ratio > 2.0),
        },
    )
    logger.info("Saved preprocessed data to %s", save_path)

    if args.convert_only:
        print(f"Data saved. To train: python train_cli.py --session {args.session_name}")
        return

    # ── Phase 6: Train ────────────────────────────────────────
    if args.loso:
        _run_loso(epochs_arr, labels_arr, subjects_arr, subjects, args)
    else:
        _run_standard(epochs_arr, labels_arr, args)


def _run_standard(epochs_arr, labels_arr, args):
    """Standard training with stratified random split."""
    logger.info(
        "Training gesture EEGNet on %d preprocessed epochs (%d classes)...",
        len(labels_arr), len(set(labels_arr.tolist())),
    )

    trainer = DeepTrainer()
    result = trainer.train_gesture(
        epochs_arr, labels_arr,
        max_epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
    )

    _print_results("STANDARD SPLIT", result)


def _run_loso(epochs_arr, labels_arr, subjects_arr, subject_names, args):
    """Leave-One-Subject-Out cross-validation with final full-data training."""
    unique_subjects = sorted(set(subjects_arr.tolist()))
    fold_results = []

    for held_out in unique_subjects:
        train_mask = subjects_arr != held_out
        val_mask = subjects_arr == held_out

        train_epochs = epochs_arr[train_mask]
        train_labels = labels_arr[train_mask]
        val_epochs = epochs_arr[val_mask]
        val_labels = labels_arr[val_mask]

        print(f"\n{'─' * 60}")
        print(f"  LOSO Fold: held-out subject = {subject_names[held_out]}")
        print(f"  Train: {len(train_labels)} epochs, Val: {len(val_labels)} epochs")
        print(f"{'─' * 60}")

        trainer = DeepTrainer()
        result = trainer.train_gesture(
            train_epochs, train_labels,
            max_epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
        )
        fold_results.append(result)

        print(f"  Subject {subject_names[held_out]}: val_acc={result.best_val_accuracy:.1%}")
        for cls, acc in result.class_accuracies.items():
            print(f"    {cls:<15} {acc:.1%}")

    avg_acc = np.mean([r.best_val_accuracy for r in fold_results])
    print(f"\n{'=' * 60}")
    print(f"  LOSO CROSS-VALIDATION SUMMARY")
    print(f"{'=' * 60}")
    for i, r in enumerate(fold_results):
        print(f"  Subject {subject_names[unique_subjects[i]]}: {r.best_val_accuracy:.1%}")
    print(f"  {'─' * 40}")
    print(f"  Mean accuracy:  {avg_acc:.1%}")
    print(f"{'=' * 60}\n")

    print("Training final model on all subjects...")
    trainer = DeepTrainer()
    result = trainer.train_gesture(
        epochs_arr, labels_arr,
        max_epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
    )
    _print_results("FINAL (all subjects)", result)


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

    # Per-class accuracy
    print(f"  Per-class accuracy:")
    for cls, acc in result.class_accuracies.items():
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        status = "✓" if acc >= 0.65 else "✗"
        print(f"    {status} {cls:<10} {bar} {acc:.1%}")
    print()

    # Confusion matrix
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

    # Classification report
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


if __name__ == "__main__":
    main()
