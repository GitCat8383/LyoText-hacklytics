#!/usr/bin/env python3
"""CLI script to train EEGNet models from collected data.

Usage:
    python train_cli.py                          # train gesture on all data
    python train_cli.py --model p300             # train P300 model
    python train_cli.py --session muse2_raw_session1  # train on specific session
    python train_cli.py --epochs 200 --lr 0.0003 # custom hyperparameters
"""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from eeg.dataset import load_all_sessions, load_epochs, list_sessions, LABEL_NAMES
from eeg.deep_trainer import DeepTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("train_cli")


def main():
    parser = argparse.ArgumentParser(description="Train EEGNet on collected EEG data")
    parser.add_argument("--model", choices=["gesture", "p300"], default="gesture")
    parser.add_argument("--session", type=str, default=None,
                        help="Train on specific session (default: all sessions)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--list", action="store_true", help="List available sessions")
    args = parser.parse_args()

    if args.list:
        sessions = list_sessions()
        if not sessions:
            print("No data sessions found. Collect data first.")
            return
        print(f"\n{'Name':<30} {'Epochs':>8} {'Classes'}")
        print("-" * 60)
        for s in sessions:
            dist = s.get("class_distribution", {})
            dist_str = ", ".join(f"{k}:{v}" for k, v in dist.items())
            print(f"{s['name']:<30} {s['n_epochs']:>8} {dist_str}")
        print()
        return

    # Load data
    if args.session:
        logger.info("Loading session: %s", args.session)
        epochs, labels = load_epochs(args.session)
    else:
        logger.info("Loading all sessions...")
        epochs, labels = load_all_sessions()

    if len(labels) == 0:
        logger.error("No data found. Collect data first.")
        return

    unique, counts = np.unique(labels, return_counts=True)
    logger.info("Loaded %d epochs, %d classes:", len(labels), len(unique))
    for idx, count in zip(unique, counts):
        name = LABEL_NAMES[idx] if idx < len(LABEL_NAMES) else str(idx)
        logger.info("  %s: %d epochs", name, count)

    if len(labels) < 20:
        logger.error("Need at least 20 epochs, got %d", len(labels))
        return

    # Train
    trainer = DeepTrainer()
    logger.info("Starting %s training (epochs=%d, lr=%g, patience=%d)...",
                args.model, args.epochs, args.lr, args.patience)

    t0 = time.time()
    if args.model == "gesture":
        result = trainer.train_gesture(
            epochs, labels,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
        )
    else:
        result = trainer.train_p300(
            epochs, labels,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
        )

    print("\n" + "=" * 50)
    print(f"  TRAINING COMPLETE — {args.model.upper()}")
    print("=" * 50)
    print(f"  Best val accuracy:  {result.best_val_accuracy:.1%}")
    print(f"  Best val loss:      {result.best_val_loss:.4f}")
    print(f"  Train accuracy:     {result.final_train_accuracy:.1%}")
    print(f"  Epochs trained:     {result.epochs_trained}")
    print(f"  Total time:         {result.total_time_sec:.1f}s")
    print(f"  Model saved to:     {result.model_path}")
    print()
    print("  Per-class accuracy:")
    for cls, acc in result.class_accuracies.items():
        print(f"    {cls:<15} {acc:.1%}")
    print("=" * 50)
    print()
    print("Model is saved. It will auto-load on next backend startup.")
    print("To reload NOW without restarting:")
    print("  curl -X POST http://localhost:8000/api/dl/models/reload")
    print()


if __name__ == "__main__":
    main()
