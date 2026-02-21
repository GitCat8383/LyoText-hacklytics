"""Blink and jaw clench detection from Muse 2 EEG channels.

Blink  -> large biphasic spikes on AF7/AF8 (frontal, near eyes)
Clench -> sustained high-RMS on TP9/TP10 (temporal, near jaw muscles)
"""

from __future__ import annotations

import logging
import threading

import numpy as np

import config
from utils.events import Event, EventType, event_bus

logger = logging.getLogger(__name__)

CH_TP9, CH_AF7, CH_AF8, CH_TP10 = 0, 1, 2, 3


class ArtifactDetector:

    def __init__(self) -> None:
        self._last_blink_time = 0.0
        self._last_clench_time = 0.0
        self._flash_active = False
        self._lock = threading.Lock()

        self._rms_window = int(0.2 * config.EEG_SAMPLE_RATE)
        self._rms_buf_tp9: list[float] = []
        self._rms_buf_tp10: list[float] = []
        self._clench_onset: float | None = None

    def set_flash_active(self, active: bool) -> None:
        with self._lock:
            self._flash_active = active

    def process_samples(self, filtered: np.ndarray, timestamps: np.ndarray) -> None:
        for i in range(len(timestamps)):
            self._check_blink(filtered[i], timestamps[i])
            self._check_clench(filtered[i], timestamps[i])

    def _check_blink(self, sample: np.ndarray, ts: float) -> None:
        af7 = abs(sample[CH_AF7])
        af8 = abs(sample[CH_AF8])
        peak = max(af7, af8)

        if peak < config.BLINK_THRESHOLD_UV:
            return

        refractory = config.BLINK_REFRACTORY_MS / 1000.0
        if ts - self._last_blink_time < refractory:
            return

        self._last_blink_time = ts
        logger.debug("Blink detected at %.3f (peak=%.1f uV)", ts, peak)
        event_bus.emit(Event(
            type=EventType.BLINK_DETECTED,
            data={"peak_uv": peak, "channel": "AF7" if af7 > af8 else "AF8"},
            timestamp=ts,
        ))

    def _check_clench(self, sample: np.ndarray, ts: float) -> None:
        with self._lock:
            if self._flash_active:
                self._clench_onset = None
                return

        self._rms_buf_tp9.append(sample[CH_TP9])
        self._rms_buf_tp10.append(sample[CH_TP10])
        if len(self._rms_buf_tp9) > self._rms_window:
            self._rms_buf_tp9.pop(0)
            self._rms_buf_tp10.pop(0)

        if len(self._rms_buf_tp9) < self._rms_window:
            return

        rms_tp9 = np.sqrt(np.mean(np.square(self._rms_buf_tp9)))
        rms_tp10 = np.sqrt(np.mean(np.square(self._rms_buf_tp10)))
        rms = max(rms_tp9, rms_tp10)

        min_duration = config.CLENCH_MIN_DURATION_MS / 1000.0

        if rms >= config.CLENCH_RMS_THRESHOLD:
            if self._clench_onset is None:
                self._clench_onset = ts
            elif ts - self._clench_onset >= min_duration:
                refractory = config.BLINK_REFRACTORY_MS / 1000.0
                if ts - self._last_clench_time >= refractory:
                    self._last_clench_time = ts
                    self._clench_onset = None
                    logger.debug("Jaw clench detected at %.3f (rms=%.1f)", ts, rms)
                    event_bus.emit(Event(
                        type=EventType.CLENCH_DETECTED,
                        data={"rms": rms},
                        timestamp=ts,
                    ))
        else:
            self._clench_onset = None


artifact_detector = ArtifactDetector()
