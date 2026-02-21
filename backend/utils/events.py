"""Thread-safe pub/sub event bus for inter-module communication."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class EventType(str, Enum):
    STIMULUS_ONSET = "stimulus_onset"
    EPOCH_READY = "epoch_ready"
    P300_RESULT = "p300_result"
    BLINK_DETECTED = "blink_detected"
    CLENCH_DETECTED = "clench_detected"
    PHRASE_CONFIRMED = "phrase_confirmed"
    PHRASE_DELETED = "phrase_deleted"
    PHRASES_UPDATED = "phrases_updated"
    CALIBRATION_PROGRESS = "calibration_progress"
    EEG_SAMPLE = "eeg_sample"
    BAND_POWER = "band_power"
    SYSTEM_STATUS = "system_status"
    FLASH_COMMAND = "flash_command"


@dataclass
class Event:
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> dict:
        return {
            "type": self.type.value,
            "data": self.data,
            "ts": self.timestamp,
        }


class EventBus:
    """Simple thread-safe pub/sub. Subscribers get their own queue."""

    def __init__(self) -> None:
        self._subscribers: dict[str, queue.Queue[Event]] = {}
        self._handlers: dict[EventType, list[Callable[[Event], None]]] = {}
        self._lock = threading.Lock()

    def subscribe(self, subscriber_id: str, maxsize: int = 256) -> queue.Queue[Event]:
        with self._lock:
            q: queue.Queue[Event] = queue.Queue(maxsize=maxsize)
            self._subscribers[subscriber_id] = q
            return q

    def unsubscribe(self, subscriber_id: str) -> None:
        with self._lock:
            self._subscribers.pop(subscriber_id, None)

    def on(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        """Register a synchronous handler for a specific event type."""
        with self._lock:
            self._handlers.setdefault(event_type, []).append(handler)

    def emit(self, event: Event) -> None:
        with self._lock:
            subscribers = list(self._subscribers.values())
            handlers = list(self._handlers.get(event.type, []))

        for q in subscribers:
            try:
                q.put_nowait(event)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass

        for handler in handlers:
            try:
                handler(event)
            except Exception:
                pass


event_bus = EventBus()
