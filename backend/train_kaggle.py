#!/usr/bin/env python3
"""Train EEGNet on the Kaggle Muse 2 facial movements dataset.

This script handles the dataset from:
    https://www.kaggle.com/datasets/muhammadatefelkaffas/eeg-muse2-motor-imagery-brain-electrical-activity

Dataset structure:
    kaggle_data/
    ├── raw_data/{subject}/{gesture}/{trial}.csv   (timestamps, TP9, AF7, AF8, TP10, Right AUX)
    └── roi_data/{subject}/{gesture}/{trial}.csv   (ROILimits_1, ROILimits_2, Value)

Gesture folders: both, left, right, eyebrows, teeth
Mapping to our labels: teeth → clench, eyebrows → blink, non-ROI → idle

Usage:
    python train_kaggle.py                           # train with default settings
    python train_kaggle.py --data-dir ./kaggle_data  # custom path
    python train_kaggle.py --all-gestures            # keep all 5 gesture classes + idle
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

from eeg.dataset import save_epochs, LABEL_MAP
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


def parse_trial(
    raw_csv: str,
    roi_csv: str | None,
    gesture: str,
    label_map: dict[str, str],
) -> tuple[list[np.ndarray], list[str]]:
    """Extract gesture and idle epochs from a single trial.

    Uses ROI annotations to extract windows where the gesture occurs,
    and non-ROI regions for idle epochs.
    """
    try:
        df = pd.read_csv(raw_csv, low_memory=False)
    except Exception as e:
        logger.warning("Failed to read %s: %s", raw_csv, e)
        return [], []

    if not all(c in df.columns for c in EEG_CHANNELS):
        logger.debug("Skipping %s — missing EEG channels (has: %s)", raw_csv, list(df.columns))
        return [], []

    eeg = df[EEG_CHANNELS].values.astype(np.float32)

    valid_mask = ~np.isnan(eeg).any(axis=1)
    eeg = eeg[valid_mask]

    if len(eeg) < WINDOW_SAMPLES:
        logger.debug("Skipping %s — too few samples (%d)", raw_csv, len(eeg))
        return [], []

    label = label_map.get(gesture)
    if label is None:
        return [], []

    epochs = []
    labels = []
    roi_ranges = []

    if roi_csv and os.path.exists(roi_csv):
        try:
            roi_df = pd.read_csv(roi_csv)
            for _, row in roi_df.iterrows():
                start = int(row["ROILimits_1"])
                end = int(row["ROILimits_2"])
                roi_ranges.append((start, end))
        except Exception as e:
            logger.warning("Failed to read ROI %s: %s", roi_csv, e)

    for start, end in roi_ranges:
        roi_len = end - start
        if roi_len >= WINDOW_SAMPLES:
            for offset in range(0, roi_len - WINDOW_SAMPLES + 1, WINDOW_SAMPLES // 2):
                s = start + offset
                e = s + WINDOW_SAMPLES
                if e <= len(eeg):
                    epochs.append(eeg[s:e].T)  # (4, 256)
                    labels.append(label)
        elif start >= 0 and end <= len(eeg):
            center = (start + end) // 2
            s = max(0, center - WINDOW_SAMPLES // 2)
            e = s + WINDOW_SAMPLES
            if e > len(eeg):
                s = len(eeg) - WINDOW_SAMPLES
                e = len(eeg)
            if s >= 0:
                epochs.append(eeg[s:e].T)
                labels.append(label)

    n_gesture = len(epochs)
    max_idle = max(n_gesture, 3)
    idle_count = 0
    stride = WINDOW_SAMPLES * 3

    for s in range(0, len(eeg) - WINDOW_SAMPLES, stride):
        if idle_count >= max_idle:
            break
        e = s + WINDOW_SAMPLES
        overlaps = any(not (e <= rs or s >= re) for rs, re in roi_ranges)
        if overlaps:
            continue
        epochs.append(eeg[s:e].T)
        labels.append("idle")
        idle_count += 1

    return epochs, labels


def main():
    parser = argparse.ArgumentParser(
        description="Train EEGNet on Kaggle Muse 2 facial movements dataset"
    )
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Directory containing raw_data/ and roi_data/")
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
    args = parser.parse_args()

    raw_dir = Path(args.data_dir) / "raw_data"
    roi_dir = Path(args.data_dir) / "roi_data"

    if not raw_dir.exists():
        logger.error("raw_data directory not found at %s", raw_dir)
        logger.error("Download from: https://www.kaggle.com/datasets/muhammadatefelkaffas/eeg-muse2-motor-imagery-brain-electrical-activity")
        logger.error("Unzip to: %s", args.data_dir)
        sys.exit(1)

    label_map = ALL_GESTURE_LABELS if args.all_gestures else GESTURE_TO_LABEL

    all_epochs = []
    all_labels = []
    stats = {"trials": 0, "skipped_gesture": 0}

    subjects = sorted([d.name for d in raw_dir.iterdir() if d.is_dir()])
    logger.info("Found %d subjects: %s", len(subjects), subjects)

    for subject in subjects:
        subject_raw = raw_dir / subject
        gestures = sorted([d.name for d in subject_raw.iterdir() if d.is_dir()])

        for gesture in gestures:
            if gesture not in label_map:
                stats["skipped_gesture"] += 1
                continue

            gesture_raw = subject_raw / gesture
            gesture_roi = roi_dir / subject / gesture

            csv_files = sorted(gesture_raw.glob("*.csv"))
            for csv_file in csv_files:
                roi_file = gesture_roi / csv_file.name if gesture_roi.exists() else None

                epochs, labels = parse_trial(
                    str(csv_file),
                    str(roi_file) if roi_file else None,
                    gesture,
                    label_map,
                )
                all_epochs.extend(epochs)
                all_labels.extend(labels)
                stats["trials"] += 1

    if not all_epochs:
        logger.error("No epochs extracted. Check the dataset directory structure.")
        sys.exit(1)

    counts = {}
    for l in all_labels:
        counts[l] = counts.get(l, 0) + 1

    print(f"\n{'=' * 50}")
    print(f"  DATASET PARSING COMPLETE")
    print(f"{'=' * 50}")
    print(f"  Subjects:         {len(subjects)}")
    print(f"  Trials processed: {stats['trials']}")
    print(f"  Total epochs:     {len(all_epochs)}")
    for cls, n in sorted(counts.items()):
        print(f"    {cls}: {n}")
    print(f"  Shape: {all_epochs[0].shape} (channels, samples)")
    print(f"{'=' * 50}\n")

    if args.all_gestures:
        extended_label_map = {**LABEL_MAP, "both": 5, "left": 6, "right": 7}
    else:
        extended_label_map = LABEL_MAP

    save_path = save_epochs(
        epochs=all_epochs,
        labels=all_labels,
        session_name=args.session_name,
        metadata={
            "source": "kaggle",
            "dataset": "muhammadatefelkaffas/eeg-muse2-motor-imagery-brain-electrical-activity",
            "all_gestures": args.all_gestures,
        },
    )
    logger.info("Saved to %s", save_path)

    if args.convert_only:
        print(f"\nData saved. To train: python train_cli.py --session {args.session_name}")
        return

    # Build numeric label arrays, skipping unknown labels
    mapped_epochs = []
    mapped_labels = []
    for i, l in enumerate(all_labels):
        if l not in extended_label_map:
            logger.warning("Unknown label '%s' at index %d, skipping", l, i)
            continue
        mapped_labels.append(extended_label_map[l])
        mapped_epochs.append(all_epochs[i])

    epochs_arr = np.array(mapped_epochs, dtype=np.float32)
    labels_arr = np.array(mapped_labels, dtype=np.int64)

    logger.info(
        "Training gesture EEGNet on %d epochs (%d classes)...",
        len(labels_arr), len(set(labels_arr.tolist())),
    )

    trainer = DeepTrainer()
    result = trainer.train_gesture(
        epochs_arr, labels_arr,
        max_epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
    )

    print(f"\n{'=' * 50}")
    print(f"  TRAINING COMPLETE — GESTURE (Kaggle data)")
    print(f"{'=' * 50}")
    print(f"  Best val accuracy:  {result.best_val_accuracy:.1%}")
    print(f"  Best val loss:      {result.best_val_loss:.4f}")
    print(f"  Train accuracy:     {result.final_train_accuracy:.1%}")
    print(f"  Epochs trained:     {result.epochs_trained}")
    print(f"  Total time:         {result.total_time_sec:.1f}s")
    print(f"  Model saved to:     {result.model_path}")
    print()
    print(f"  Per-class accuracy:")
    for cls, acc in result.class_accuracies.items():
        print(f"    {cls:<15} {acc:.1%}")
    print(f"{'=' * 50}")
    print()
    print(f"Model saved. To load into running backend:")
    print(f"  curl -X POST http://localhost:8000/api/dl/models/reload")
    print()


if __name__ == "__main__":
    main()
