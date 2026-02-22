"""Data collection manager for EEGNet training.

Runs guided collection sessions where the user performs gestures
(blink, clench, idle) on cue, and the system records labeled EEG
epochs for later training.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import config
from database.store import redis_store
from eeg.dataset import save_epochs, LABEL_MAP
from utils.events import Event, EventType, event_bus

logger = logging.getLogger(__name__)

GESTURE_WINDOW_SEC = 1.0
IDLE_WINDOW_SEC = 1.0


@dataclass
class CollectionSession:
    name: str
    gesture_types: list[str]
    trials_per_gesture: int
    current_gesture: str = ""
    current_trial: int = 0
    total_trials: int = 0
    collected_epochs: list[np.ndarray] = field(default_factory=list)
    collected_labels: list[str] = field(default_factory=list)
    running: bool = False
    paused: bool = False


class DataCollector:
    """Manages data collection sessions for training EEGNet."""

    def __init__(self) -> None:
        self._session: CollectionSession | None = None
        self._lock = threading.Lock()

    @property
    def is_collecting(self) -> bool:
        return self._session is not None and self._session.running

    @property
    def session_status(self) -> dict[str, Any] | None:
        if self._session is None:
            return None
        s = self._session
        return {
            "name": s.name,
            "running": s.running,
            "paused": s.paused,
            "current_gesture": s.current_gesture,
            "current_trial": s.current_trial,
            "total_trials": s.total_trials,
            "collected": len(s.collected_epochs),
            "gesture_types": s.gesture_types,
            "trials_per_gesture": s.trials_per_gesture,
        }

    def start_session(
        self,
        name: str,
        gesture_types: list[str] | None = None,
        trials_per_gesture: int = 30,
    ) -> dict[str, Any]:
        """Start a new guided collection session.

        Args:
            name: session identifier (used as filename)
            gesture_types: list of gestures to collect (default: idle, blink, clench)
            trials_per_gesture: how many epochs per gesture class
        """
        if self._session and self._session.running:
            raise RuntimeError("A collection session is already running")

        if gesture_types is None:
            gesture_types = ["idle", "blink", "clench"]

        for g in gesture_types:
            if g not in LABEL_MAP:
                raise ValueError(f"Unknown gesture type: {g}")

        total = len(gesture_types) * trials_per_gesture
        session = CollectionSession(
            name=name,
            gesture_types=gesture_types,
            trials_per_gesture=trials_per_gesture,
            total_trials=total,
        )
        self._session = session

        threading.Thread(
            target=self._run_session, args=(session,), daemon=True
        ).start()

        return self.session_status

    def stop_session(self) -> dict[str, Any]:
        """Stop the current session and save whatever has been collected."""
        if not self._session:
            raise RuntimeError("No active session")

        session = self._session
        session.running = False

        result = self._save_session(session)
        self._session = None
        return result

    def pause_session(self) -> None:
        if self._session:
            self._session.paused = True

    def resume_session(self) -> None:
        if self._session:
            self._session.paused = False

    def add_manual_epoch(self, label: str) -> dict[str, Any]:
        """Manually capture current 1-second window with a label."""
        if label not in LABEL_MAP:
            raise ValueError(f"Unknown label: {label}")

        epoch = self._capture_window(GESTURE_WINDOW_SEC)
        if epoch is None:
            raise RuntimeError("Not enough EEG data in buffer")

        with self._lock:
            if self._session is None:
                self._session = CollectionSession(
                    name=f"manual_{int(time.time())}",
                    gesture_types=[label],
                    trials_per_gesture=0,
                    running=False,
                )

            self._session.collected_epochs.append(epoch)
            self._session.collected_labels.append(label)

        return {
            "label": label,
            "shape": list(epoch.shape),
            "total_collected": len(self._session.collected_epochs),
        }

    def save_manual(self, session_name: str | None = None) -> dict[str, Any]:
        """Save manually collected epochs."""
        if not self._session or not self._session.collected_epochs:
            raise RuntimeError("No epochs to save")

        name = session_name or self._session.name
        result = self._save_session(self._session, name)
        self._session = None
        return result

    # ── Internal ──────────────────────────────────────────────

    def _run_session(self, session: CollectionSession) -> None:
        """Guided collection: cycle through gesture types with cues."""
        session.running = True
        trial_idx = 0

        logger.info(
            "Collection session '%s' started: %s, %d trials each",
            session.name, session.gesture_types, session.trials_per_gesture,
        )

        event_bus.emit(Event(
            type=EventType.SYSTEM_STATUS,
            data={
                "status": "collection_started",
                "session": session.name,
                "gesture_types": session.gesture_types,
                "total_trials": session.total_trials,
            },
        ))

        # Randomize trial order for balanced collection
        trial_plan = []
        for gesture in session.gesture_types:
            trial_plan.extend([gesture] * session.trials_per_gesture)
        rng = np.random.default_rng(42)
        rng.shuffle(trial_plan)

        # 3-second countdown
        for countdown in [3, 2, 1]:
            if not session.running:
                break
            event_bus.emit(Event(
                type=EventType.CALIBRATION_PROGRESS,
                data={"countdown": countdown, "message": f"Starting in {countdown}..."},
            ))
            time.sleep(1.0)

        for gesture in trial_plan:
            if not session.running:
                break

            while session.paused:
                time.sleep(0.1)
                if not session.running:
                    break

            trial_idx += 1
            session.current_gesture = gesture
            session.current_trial = trial_idx

            # Announce the cue
            event_bus.emit(Event(
                type=EventType.CALIBRATION_PROGRESS,
                data={
                    "phase": "cue",
                    "gesture": gesture,
                    "trial": trial_idx,
                    "total": session.total_trials,
                    "message": f"Perform: {gesture.upper()}",
                },
            ))

            # Short prep period
            time.sleep(0.5)

            # Record window
            event_bus.emit(Event(
                type=EventType.CALIBRATION_PROGRESS,
                data={
                    "phase": "recording",
                    "gesture": gesture,
                    "trial": trial_idx,
                    "total": session.total_trials,
                    "message": f"Recording {gesture}...",
                },
            ))

            window_sec = GESTURE_WINDOW_SEC if gesture != "idle" else IDLE_WINDOW_SEC
            time.sleep(window_sec)

            epoch = self._capture_window(window_sec)
            if epoch is not None:
                with self._lock:
                    session.collected_epochs.append(epoch)
                    session.collected_labels.append(gesture)

            # Rest between trials
            event_bus.emit(Event(
                type=EventType.CALIBRATION_PROGRESS,
                data={
                    "phase": "rest",
                    "trial": trial_idx,
                    "total": session.total_trials,
                    "message": "Rest...",
                },
            ))
            time.sleep(1.0)

        # Session complete — save
        if session.collected_epochs:
            result = self._save_session(session)
            event_bus.emit(Event(
                type=EventType.SYSTEM_STATUS,
                data={
                    "status": "collection_complete",
                    "session": session.name,
                    **result,
                },
            ))
        else:
            event_bus.emit(Event(
                type=EventType.SYSTEM_STATUS,
                data={"status": "collection_empty", "session": session.name},
            ))

        session.running = False

    def _capture_window(self, window_sec: float) -> np.ndarray | None:
        """Extract the latest `window_sec` of RAW (unfiltered) EEG from Redis.

        Raw data preserves blink spikes and jaw-clench EMG artifacts that the
        bandpass filter would otherwise attenuate — critical for gesture training.
        """
        try:
            raw_samples = redis_store.get_recent_raw(seconds=window_sec)
        except Exception:
            logger.debug("Redis raw read failed")
            return None

        n_needed = int(window_sec * config.EEG_SAMPLE_RATE * 0.8)
        if len(raw_samples) < n_needed:
            return None

        data = np.array([
            [s["tp9"], s["af7"], s["af8"], s["tp10"]]
            for s in raw_samples
        ], dtype=np.float32)

        # Ensure consistent length
        target_len = int(window_sec * config.EEG_SAMPLE_RATE)
        if len(data) > target_len:
            data = data[-target_len:]
        elif len(data) < target_len:
            pad = target_len - len(data)
            data = np.pad(data, ((pad, 0), (0, 0)), mode="edge")

        # Return shape (n_channels, n_samples) = (4, 256) for 1 second
        return data.T

    def _save_session(
        self,
        session: CollectionSession,
        name: str | None = None,
    ) -> dict[str, Any]:
        name = name or session.name
        path = save_epochs(
            epochs=session.collected_epochs,
            labels=session.collected_labels,
            session_name=name,
            metadata={
                "gesture_types": session.gesture_types,
                "trials_per_gesture": session.trials_per_gesture,
            },
        )

        n_per_class = {}
        for label in session.collected_labels:
            n_per_class[label] = n_per_class.get(label, 0) + 1

        return {
            "saved_to": str(path),
            "total_epochs": len(session.collected_epochs),
            "class_distribution": n_per_class,
        }


data_collector = DataCollector()
