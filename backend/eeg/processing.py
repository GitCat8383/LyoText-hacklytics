"""Real-time bandpass filtering and stimulus-locked epoching."""

from __future__ import annotations

import collections
import logging
import threading
import time

import numpy as np
from scipy.signal import butter, sosfilt

import config
from utils.events import Event, EventType, event_bus

logger = logging.getLogger(__name__)


def _design_bandpass() -> np.ndarray:
    nyq = config.EEG_SAMPLE_RATE / 2.0
    low = config.BANDPASS_LOW / nyq
    high = config.BANDPASS_HIGH / nyq
    return butter(config.BANDPASS_ORDER, [low, high], btype="band", output="sos")


class SignalProcessor:

    def __init__(self) -> None:
        self._sos = _design_bandpass()
        self._zi: list[np.ndarray | None] = [None] * config.NUM_CHANNELS
        self._lock = threading.Lock()

        buf_len = config.EEG_SAMPLE_RATE * 2
        self._buffer = np.zeros((buf_len, config.NUM_CHANNELS))
        self._ts_buffer = np.zeros(buf_len)
        self._buf_idx = 0
        self._buf_count = 0

        self._markers: collections.deque[tuple[float, int]] = collections.deque(maxlen=100)
        event_bus.on(EventType.STIMULUS_ONSET, self._on_stimulus)

    def _on_stimulus(self, event: Event) -> None:
        ts = event.timestamp
        phrase_idx = event.data.get("phrase_index", -1)
        self._markers.append((ts, phrase_idx))

    def process_chunk(self, samples: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        filtered = np.empty_like(samples)

        with self._lock:
            for ch in range(config.NUM_CHANNELS):
                if self._zi[ch] is None:
                    from scipy.signal import sosfilt_zi
                    self._zi[ch] = sosfilt_zi(self._sos) * samples[0, ch]

                filtered[:, ch], self._zi[ch] = sosfilt(
                    self._sos, samples[:, ch], zi=self._zi[ch]
                )

            buf_len = len(self._buffer)
            n = len(timestamps)
            for i in range(n):
                idx = self._buf_idx % buf_len
                self._buffer[idx] = filtered[i]
                self._ts_buffer[idx] = timestamps[i]
                self._buf_idx += 1
                self._buf_count = min(self._buf_count + 1, buf_len)

        return filtered

    def try_extract_epochs(self) -> list[tuple[np.ndarray, int]]:
        if not self._markers:
            return []

        now = time.time()
        ready = []
        remaining: list[tuple[float, int]] = []

        for marker_ts, phrase_idx in self._markers:
            epoch_end = marker_ts + config.EPOCH_TMAX
            if now >= epoch_end + 0.05:
                epoch = self._extract_epoch(marker_ts)
                if epoch is not None:
                    ready.append((epoch, phrase_idx))
            else:
                remaining.append((marker_ts, phrase_idx))

        self._markers.clear()
        self._markers.extend(remaining)

        for epoch, phrase_idx in ready:
            event_bus.emit(Event(
                type=EventType.EPOCH_READY,
                data={"phrase_index": phrase_idx, "shape": list(epoch.shape)},
            ))

        return ready

    def _extract_epoch(self, marker_ts: float) -> np.ndarray | None:
        with self._lock:
            if self._buf_count == 0:
                return None

            buf_len = len(self._buffer)
            valid_start = max(0, self._buf_idx - self._buf_count)
            indices = [i % buf_len for i in range(valid_start, self._buf_idx)]

            ts_arr = self._ts_buffer[indices]
            data_arr = self._buffer[indices]

        epoch_start = marker_ts + config.EPOCH_TMIN
        epoch_end = marker_ts + config.EPOCH_TMAX

        mask = (ts_arr >= epoch_start) & (ts_arr <= epoch_end)
        epoch_data = data_arr[mask]

        if len(epoch_data) < config.EPOCH_SAMPLES * 0.8:
            return None

        epoch = epoch_data.T

        pre_stim_samples = int(abs(config.EPOCH_TMIN) * config.EEG_SAMPLE_RATE)
        if pre_stim_samples > 0 and epoch.shape[1] > pre_stim_samples:
            baseline = epoch[:, :pre_stim_samples].mean(axis=1, keepdims=True)
            epoch = epoch - baseline

        if epoch.shape[1] < config.EPOCH_SAMPLES:
            pad_width = config.EPOCH_SAMPLES - epoch.shape[1]
            epoch = np.pad(epoch, ((0, 0), (0, pad_width)), mode="edge")
        elif epoch.shape[1] > config.EPOCH_SAMPLES:
            epoch = epoch[:, : config.EPOCH_SAMPLES]

        return epoch


signal_processor = SignalProcessor()
