"""Muse 2 EEG streaming via Lab Streaming Layer (LSL).

Launches muselsl in a subprocess, reads samples via pylsl in a background
thread, and pushes data to Redis + the internal event bus.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

import config
from database.store import redis_store
from utils.events import Event, EventType, event_bus

if TYPE_CHECKING:
    from queue import Queue

logger = logging.getLogger(__name__)

CH_TP9, CH_AF7, CH_AF8, CH_TP10 = 0, 1, 2, 3


class EEGStream:
    """Manages the Muse 2 -> LSL -> Python pipeline."""

    def __init__(self) -> None:
        self._muse_proc: subprocess.Popen | None = None
        self._inlet = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._sample_buffer: Queue | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self, sample_buffer: Queue | None = None) -> None:
        self._sample_buffer = sample_buffer
        self._running = True

        if config.SIMULATE_EEG:
            logger.info("Starting simulated EEG stream")
            self._thread = threading.Thread(
                target=self._simulate_loop, daemon=True
            )
        else:
            self._start_muselsl()
            self._connect_inlet()
            self._thread = threading.Thread(
                target=self._read_loop, daemon=True
            )

        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._muse_proc:
            self._muse_proc.terminate()
            self._muse_proc = None

    def _start_muselsl(self) -> None:
        logger.info("Launching muselsl stream subprocess...")
        self._muse_proc = subprocess.Popen(
            [sys.executable, "-m", "muselsl", "stream"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(3)
        if self._muse_proc.poll() is not None:
            stderr = self._muse_proc.stderr.read().decode() if self._muse_proc.stderr else ""
            raise RuntimeError(f"muselsl stream failed to start: {stderr}")
        logger.info("muselsl stream subprocess started (pid=%d)", self._muse_proc.pid)

    def _connect_inlet(self) -> None:
        import pylsl

        logger.info("Resolving LSL stream 'Muse'...")
        streams = pylsl.resolve_byprop("type", "EEG", timeout=10)
        if not streams:
            raise RuntimeError(
                "No EEG LSL stream found. Is muselsl running and Muse 2 connected?"
            )
        self._inlet = pylsl.StreamInlet(streams[0], max_chunklen=12)
        logger.info("Connected to LSL EEG stream: %s", streams[0].name())

    def _read_loop(self) -> None:
        while self._running:
            try:
                samples, timestamps = self._inlet.pull_chunk(
                    timeout=0.05, max_samples=32
                )
                if not timestamps:
                    continue
                self._distribute(np.array(samples), np.array(timestamps))
            except Exception:
                logger.exception("Error in EEG read loop")
                time.sleep(0.1)

    def _simulate_loop(self) -> None:
        """Generate synthetic EEG with realistic artifacts and P300 responses."""
        rng = np.random.default_rng(42)
        sample_interval = 1.0 / config.EEG_SAMPLE_RATE
        chunk_size = 12

        self._sim_stim_times: list[tuple[float, int]] = []
        event_bus.on(EventType.STIMULUS_ONSET, self._on_sim_stimulus)

        sim_target = rng.integers(0, config.NUM_PHRASES)
        logger.info("Simulation target phrase index: %d", sim_target)

        blink_timer = time.time() + rng.uniform(8, 15)
        clench_timer = time.time() + rng.uniform(25, 40)

        while self._running:
            now = time.time()
            timestamps_np = np.array(
                [now + i * sample_interval for i in range(chunk_size)]
            )

            noise = rng.normal(0, 10, (chunk_size, config.NUM_CHANNELS))
            alpha = 5 * np.sin(
                2 * np.pi * 10 * timestamps_np[:, None]
                * np.ones((1, config.NUM_CHANNELS))
            )
            samples_np = noise + alpha

            for stim_ts, phrase_idx in list(self._sim_stim_times):
                for i, ts in enumerate(timestamps_np):
                    latency = ts - stim_ts
                    if 0.25 < latency < 0.45 and phrase_idx == sim_target:
                        p300_amp = 15 * np.exp(-((latency - 0.35) ** 2) / (2 * 0.03**2))
                        samples_np[i, CH_TP9] += p300_amp
                        samples_np[i, CH_TP10] += p300_amp
                if now - stim_ts > 1.0:
                    self._sim_stim_times.remove((stim_ts, phrase_idx))

            if now >= blink_timer:
                for i in range(min(3, chunk_size)):
                    samples_np[i, CH_AF7] += rng.uniform(180, 300)
                    samples_np[i, CH_AF8] += rng.uniform(150, 280)
                blink_timer = now + rng.uniform(8, 15)

            if now >= clench_timer:
                for i in range(chunk_size):
                    samples_np[i, CH_TP9] += rng.uniform(60, 90)
                    samples_np[i, CH_TP10] += rng.uniform(60, 90)
                clench_timer = now + rng.uniform(30, 50)

            self._distribute(samples_np, timestamps_np)
            time.sleep(chunk_size * sample_interval)

    def _on_sim_stimulus(self, event: Event) -> None:
        if hasattr(self, "_sim_stim_times"):
            self._sim_stim_times.append(
                (event.timestamp, event.data.get("phrase_index", -1))
            )

    def _distribute(self, samples: np.ndarray, timestamps: np.ndarray) -> None:
        try:
            redis_store.push_raw(samples, timestamps)
        except Exception:
            logger.debug("Redis push failed (non-critical)")

        if self._sample_buffer is not None:
            try:
                self._sample_buffer.put_nowait((samples, timestamps))
            except Exception:
                pass

        for i in range(len(timestamps)):
            event_bus.emit(Event(
                type=EventType.EEG_SAMPLE,
                data={
                    "tp9": float(samples[i, CH_TP9]),
                    "af7": float(samples[i, CH_AF7]),
                    "af8": float(samples[i, CH_AF8]),
                    "tp10": float(samples[i, CH_TP10]),
                },
                timestamp=float(timestamps[i]),
            ))


eeg_stream = EEGStream()
