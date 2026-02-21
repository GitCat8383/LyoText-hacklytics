"""Real-time bandpass filtering and stimulus-locked epoching."""

from __future__ import annotations

import collections
import logging
import threading
import time

import numpy as np
from scipy.signal import butter, sosfilt, welch

import config
from utils.events import Event, EventType, event_bus

# EEG frequency bands (Hz)
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

BAND_POWER_INTERVAL = 0.5  # seconds between band-power emissions

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

        self._last_band_power_ts = 0.0
        threading.Thread(target=self._band_power_loop, daemon=True).start()

    def _on_stimulus(self, event: Event) -> None:
        ts = event.timestamp
        phrase_idx = event.data.get("phrase_index", -1)
        self._markers.append((ts, phrase_idx))

    def process_chunk(self, samples: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        samples = samples[:, : config.NUM_CHANNELS]
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

    def _band_power_loop(self) -> None:
        """Emit band power every BAND_POWER_INTERVAL seconds."""
        while True:
            time.sleep(BAND_POWER_INTERVAL)
            try:
                self._emit_band_power()
            except Exception:
                logger.warning("Band power computation failed", exc_info=True)

    def compute_band_power(self) -> dict:
        """Compute band power from the ring buffer. Returns dict with bands + per_channel."""
        with self._lock:
            n = min(self._buf_count, len(self._buffer))
            if n < config.EEG_SAMPLE_RATE:
                return {"bands": {}, "per_channel": {}, "error": "not enough data yet"}
            indices = [(self._buf_idx - n + i) % len(self._buffer) for i in range(n)]
            data = self._buffer[indices].copy()

        fs = config.EEG_SAMPLE_RATE
        nperseg = min(256, n)
        band_data: dict[str, float] = {}

        _integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

        for band_name, (lo, hi) in BANDS.items():
            powers = []
            for ch in range(config.NUM_CHANNELS):
                f, pxx = welch(data[:, ch], fs=fs, nperseg=nperseg)
                mask = (f >= lo) & (f <= hi)
                if mask.any():
                    powers.append(float(_integrate(pxx[mask], f[mask])))
            band_data[band_name] = float(np.mean(powers)) if powers else 0.0

        per_channel: dict[str, dict[str, float]] = {}
        for ch_idx, ch_name in enumerate(config.EEG_CHANNELS):
            f, pxx = welch(data[:, ch_idx], fs=fs, nperseg=nperseg)
            ch_bands: dict[str, float] = {}
            for band_name, (lo, hi) in BANDS.items():
                mask = (f >= lo) & (f <= hi)
                ch_bands[band_name] = float(_integrate(pxx[mask], f[mask])) if mask.any() else 0.0
            per_channel[ch_name] = ch_bands

        return {"bands": band_data, "per_channel": per_channel}

    def _emit_band_power(self) -> None:
        result = self.compute_band_power()
        if "error" in result:
            return
        event_bus.emit(Event(
            type=EventType.BAND_POWER,
            data=result,
        ))


signal_processor = SignalProcessor()
