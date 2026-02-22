#!/usr/bin/env python3
"""Auto-collect labeled EEG data using the threshold-based artifact detector.

Runs alongside the backend. Listens for blink/clench events from the existing
rule-based detector and captures the raw EEG window as labeled training data.
Idle windows are captured at random intervals between events.

This "teacher-student" approach bootstraps the EEGNet from the rule-based system.

Usage:
    python auto_collect.py                     # collect until Ctrl+C
    python auto_collect.py --min-epochs 100    # collect at least 100 epochs
    python auto_collect.py --name my_session   # custom session name
"""

import argparse
import logging
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import config
from database.store import redis_store
from eeg.dataset import save_epochs, LABEL_MAP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("auto_collect")

BLINK_THRESHOLD = config.BLINK_THRESHOLD_UV
CLENCH_RMS_THRESHOLD = config.CLENCH_RMS_THRESHOLD
WINDOW_SEC = 1.0
IDLE_CAPTURE_INTERVAL = 5.0
REFRACTORY_SEC = 1.5


def capture_raw_window(seconds: float = 1.0) -> np.ndarray | None:
    """Grab the latest `seconds` of raw EEG from Redis."""
    try:
        samples = redis_store.get_recent_raw(seconds=seconds)
    except Exception:
        return None

    needed = int(seconds * config.EEG_SAMPLE_RATE * 0.7)
    if len(samples) < needed:
        return None

    data = np.array([
        [s["tp9"], s["af7"], s["af8"], s["tp10"]]
        for s in samples
    ], dtype=np.float32)

    target = int(seconds * config.EEG_SAMPLE_RATE)
    if len(data) > target:
        data = data[-target:]
    elif len(data) < target:
        pad = target - len(data)
        data = np.pad(data, ((pad, 0), (0, 0)), mode="edge")

    return data.T  # (4, 256)


def detect_blink(window: np.ndarray) -> bool:
    """Check if window contains a blink (large spike on AF7/AF8)."""
    af7_peak = np.max(np.abs(window[1]))  # AF7
    af8_peak = np.max(np.abs(window[2]))  # AF8
    return max(af7_peak, af8_peak) > BLINK_THRESHOLD


def detect_clench(window: np.ndarray) -> bool:
    """Check if window contains a clench (high RMS on TP9/TP10 in any 200ms sub-window)."""
    sub_win = int(0.2 * config.EEG_SAMPLE_RATE)  # ~51 samples
    n = window.shape[1]
    for start in range(0, n - sub_win, sub_win // 2):
        seg = window[:, start:start + sub_win]
        rms_tp9 = np.sqrt(np.mean(seg[0] ** 2))
        rms_tp10 = np.sqrt(np.mean(seg[3] ** 2))
        if max(rms_tp9, rms_tp10) > CLENCH_RMS_THRESHOLD:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Auto-collect labeled EEG data")
    parser.add_argument("--name", type=str, default=None,
                        help="Session name (default: auto_<timestamp>)")
    parser.add_argument("--min-epochs", type=int, default=0,
                        help="Minimum epochs to collect before allowing stop (0 = unlimited)")
    parser.add_argument("--max-epochs", type=int, default=500,
                        help="Maximum epochs to collect")
    parser.add_argument("--idle-ratio", type=float, default=1.0,
                        help="Ratio of idle samples per event sample (default: 1.0)")
    args = parser.parse_args()

    session_name = args.name or f"auto_{int(time.time())}"

    if not redis_store.ping():
        logger.error("Redis not available. Start Redis and the backend first.")
        return

    # Wait for EEG data
    logger.info("Waiting for EEG data in Redis...")
    for _ in range(30):
        w = capture_raw_window(0.5)
        if w is not None:
            break
        time.sleep(1)
    else:
        logger.error("No EEG data found after 30s. Is the backend running?")
        return

    logger.info("EEG data detected. Starting auto-collection '%s'", session_name)
    logger.info("Thresholds: blink=%.0f μV, clench_rms=%.0f", BLINK_THRESHOLD, CLENCH_RMS_THRESHOLD)
    logger.info("Press Ctrl+C to stop and save.\n")

    epochs: list[np.ndarray] = []
    labels: list[str] = []
    counts = {"idle": 0, "blink": 0, "clench": 0}

    running = True
    def _sigint(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, _sigint)

    last_event_time = 0.0
    last_idle_time = 0.0
    check_interval = 0.1  # 100ms polling

    while running:
        time.sleep(check_interval)
        now = time.time()

        if len(epochs) >= args.max_epochs:
            logger.info("Reached max epochs (%d). Stopping.", args.max_epochs)
            break

        # Check for events (refractory period)
        if now - last_event_time < REFRACTORY_SEC:
            continue

        window = capture_raw_window(WINDOW_SEC)
        if window is None:
            continue

        detected = None
        if detect_blink(window):
            detected = "blink"
        elif detect_clench(window):
            detected = "clench"

        if detected:
            epochs.append(window)
            labels.append(detected)
            counts[detected] += 1
            last_event_time = now
            logger.info(
                "  %-8s captured | blink:%d  clench:%d  idle:%d  (total:%d)",
                detected.upper(), counts["blink"], counts["clench"],
                counts["idle"], len(epochs),
            )

        # Capture idle at intervals (balanced with events)
        elif now - last_idle_time >= IDLE_CAPTURE_INTERVAL:
            max_idle = int(max(counts["blink"], counts["clench"]) * args.idle_ratio) + 5
            if counts["idle"] < max_idle:
                epochs.append(window)
                labels.append("idle")
                counts["idle"] += 1
                last_idle_time = now
                if counts["idle"] % 5 == 0:
                    logger.info(
                        "  %-8s captured | blink:%d  clench:%d  idle:%d  (total:%d)",
                        "IDLE", counts["blink"], counts["clench"],
                        counts["idle"], len(epochs),
                    )

    # Save
    print()
    if not epochs:
        logger.warning("No epochs collected.")
        return

    if args.min_epochs > 0 and len(epochs) < args.min_epochs:
        logger.warning(
            "Only %d epochs collected (minimum %d). Saving anyway.",
            len(epochs), args.min_epochs,
        )

    path = save_epochs(epochs, labels, session_name)

    print(f"\n{'=' * 50}")
    print(f"  AUTO-COLLECTION COMPLETE")
    print(f"{'=' * 50}")
    print(f"  Session:   {session_name}")
    print(f"  Total:     {len(epochs)} epochs")
    print(f"  Blink:     {counts['blink']}")
    print(f"  Clench:    {counts['clench']}")
    print(f"  Idle:      {counts['idle']}")
    print(f"  Saved to:  {path}")
    print(f"{'=' * 50}")
    print(f"\nTo train: python train_cli.py --session {session_name}")
    print()


if __name__ == "__main__":
    main()
