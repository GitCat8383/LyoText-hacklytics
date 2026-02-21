"""Redis-backed real-time storage for EEG signals, epochs, and events."""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import redis

import config


class RedisStore:

    def __init__(self, url: str | None = None) -> None:
        self._url = url or config.REDIS_URL
        self._client: redis.Redis | None = None

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.Redis.from_url(self._url, decode_responses=True)
        return self._client

    def ping(self) -> bool:
        try:
            return self.client.ping()
        except redis.ConnectionError:
            return False

    # ── Raw EEG ───────────────────────────────────────────────

    def push_raw(self, samples: np.ndarray, timestamps: np.ndarray) -> None:
        """Push a chunk of raw EEG samples to Redis Stream.

        samples: shape (n_samples, n_channels)
        timestamps: shape (n_samples,)
        """
        pipe = self.client.pipeline()
        for i in range(len(timestamps)):
            entry = {
                "ts": str(timestamps[i]),
                "tp9": str(samples[i, 0]),
                "af7": str(samples[i, 1]),
                "af8": str(samples[i, 2]),
                "tp10": str(samples[i, 3]),
            }
            pipe.xadd(
                config.REDIS_RAW_STREAM,
                entry,
                maxlen=config.REDIS_RAW_MAXLEN,
                approximate=True,
            )
        pipe.execute()

    def get_recent_raw(self, seconds: float = 1.0) -> list[dict[str, Any]]:
        cutoff_ms = int((time.time() - seconds) * 1000)
        entries = self.client.xrange(
            config.REDIS_RAW_STREAM,
            min=f"{cutoff_ms}-0",
            max="+",
        )
        return [
            {
                "ts": float(data["ts"]),
                "tp9": float(data["tp9"]),
                "af7": float(data["af7"]),
                "af8": float(data["af8"]),
                "tp10": float(data["tp10"]),
            }
            for _id, data in entries
        ]

    def get_raw_at_second(self, offset_sec: float) -> dict[str, Any]:
        """Read ~1 second of EEG data centered at `offset_sec` seconds ago.

        Args:
            offset_sec: How many seconds ago to read from (e.g. 10 = 10 seconds ago).

        Returns:
            Dict with metadata and list of samples within that 1-second window.
        """
        now = time.time()
        window_start = now - offset_sec
        window_end = window_start + 1.0

        start_ms = int(window_start * 1000)
        end_ms = int(window_end * 1000)

        entries = self.client.xrange(
            config.REDIS_RAW_STREAM,
            min=f"{start_ms}-0",
            max=f"{end_ms}-0",
        )

        samples = [
            {
                "ts": float(data["ts"]),
                "tp9": float(data["tp9"]),
                "af7": float(data["af7"]),
                "af8": float(data["af8"]),
                "tp10": float(data["tp10"]),
            }
            for _id, data in entries
        ]

        return {
            "offset_sec": offset_sec,
            "window_start": window_start,
            "window_end": window_end,
            "sample_count": len(samples),
            "expected_samples": config.EEG_SAMPLE_RATE,
            "samples": samples,
        }

    # ── Epochs ────────────────────────────────────────────────

    def push_epoch(
        self, epoch: np.ndarray, label: str, confidence: float, phrase_index: int
    ) -> None:
        entry = {
            "ts": str(time.time()),
            "label": label,
            "confidence": str(confidence),
            "phrase_index": str(phrase_index),
            "shape": json.dumps(list(epoch.shape)),
        }
        self.client.xadd(
            config.REDIS_EPOCH_STREAM,
            entry,
            maxlen=500,
            approximate=True,
        )

    # ── Events ────────────────────────────────────────────────

    def push_event(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        entry = {
            "ts": str(time.time()),
            "type": event_type,
            "data": json.dumps(data or {}),
        }
        self.client.xadd(
            config.REDIS_EVENT_STREAM,
            entry,
            maxlen=1000,
            approximate=True,
        )

    def get_recent_events(self, seconds: float = 60.0) -> list[dict[str, Any]]:
        cutoff_ms = int((time.time() - seconds) * 1000)
        entries = self.client.xrange(
            config.REDIS_EVENT_STREAM,
            min=f"{cutoff_ms}-0",
            max="+",
        )
        return [
            {
                "ts": float(data["ts"]),
                "type": data["type"],
                "data": json.loads(data["data"]),
            }
            for _id, data in entries
        ]

    def flush(self) -> None:
        self.client.delete(
            config.REDIS_RAW_STREAM,
            config.REDIS_EPOCH_STREAM,
            config.REDIS_EVENT_STREAM,
        )


redis_store = RedisStore()
