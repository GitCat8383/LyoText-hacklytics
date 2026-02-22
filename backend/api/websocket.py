"""WebSocket manager for broadcasting real-time events to React frontend."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from utils.events import Event, EventType, event_bus

logger = logging.getLogger(__name__)


_EEG_DOWNSAMPLE = 8  # forward 1 in every N samples → 256Hz / 8 = 32Hz to frontend


class WebSocketManager:

    def __init__(self) -> None:
        self._clients: list[WebSocket] = []
        self._eeg_clients: list[WebSocket] = []
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._eeg_sample_counter = 0

        event_bus.on(EventType.BLINK_DETECTED, self._on_event)
        event_bus.on(EventType.CLENCH_DETECTED, self._on_event)
        event_bus.on(EventType.P300_RESULT, self._on_event)
        event_bus.on(EventType.PHRASES_UPDATED, self._on_event)
        event_bus.on(EventType.PHRASE_CONFIRMED, self._on_event)
        event_bus.on(EventType.PHRASE_DELETED, self._on_event)
        event_bus.on(EventType.CALIBRATION_PROGRESS, self._on_event)
        event_bus.on(EventType.SYSTEM_STATUS, self._on_event)
        event_bus.on(EventType.BAND_POWER, self._on_event)
        event_bus.on(EventType.GESTURE_PREDICTION, self._on_event)
        event_bus.on(EventType.HIGHLIGHT_CHANGED, self._on_event)
        event_bus.on(EventType.SELECTION_CONFIRMED, self._on_event)
        event_bus.on(EventType.WARMUP_STATUS, self._on_event)
        event_bus.on(EventType.CALIBRATION_STATUS, self._on_event)
        event_bus.on(EventType.SELECTION_EXECUTED, self._on_event)
        event_bus.on(EventType.WORDS_UPDATED, self._on_event)
        event_bus.on(EventType.WORD_SELECTED, self._on_event)
        event_bus.on(EventType.SENTENCE_CLEARED, self._on_event)
        event_bus.on(EventType.SESSION_STOPPED, self._on_event)
        event_bus.on(EventType.EEG_SAMPLE, self._on_eeg_sample)

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def _on_event(self, event: Event) -> None:
        if self._loop is None or self._loop.is_closed():
            return
        try:
            self._loop.call_soon_threadsafe(
                asyncio.ensure_future,
                self._broadcast(event.to_json()),
            )
        except RuntimeError:
            pass

    def _on_eeg_sample(self, event: Event) -> None:
        if self._loop is None or self._loop.is_closed():
            return
        if not self._eeg_clients:
            return
        self._eeg_sample_counter += 1
        if self._eeg_sample_counter % _EEG_DOWNSAMPLE != 0:
            return
        try:
            self._loop.call_soon_threadsafe(
                asyncio.ensure_future,
                self.send_eeg_sample(event.data),
            )
        except RuntimeError:
            pass

    async def connect(self, ws: WebSocket, stream_eeg: bool = False) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.append(ws)
            if stream_eeg:
                self._eeg_clients.append(ws)
        logger.info("WebSocket client connected (stream_eeg=%s)", stream_eeg)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            if ws in self._clients:
                self._clients.remove(ws)
            if ws in self._eeg_clients:
                self._eeg_clients.remove(ws)

    async def _broadcast(self, data: dict[str, Any]) -> None:
        async with self._lock:
            targets = list(self._clients)
        dead: list[WebSocket] = []
        message = json.dumps(data)
        for ws in targets:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws)

    async def send_eeg_sample(self, data: dict[str, Any]) -> None:
        async with self._lock:
            targets = list(self._eeg_clients)
        if not targets:
            return
        message = json.dumps({"type": "eeg_sample", "data": data})
        dead: list[WebSocket] = []
        for ws in targets:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws)

    async def handle(self, ws: WebSocket) -> None:
        await self.connect(ws, stream_eeg=False)
        try:
            while True:
                msg = await ws.receive_text()
                try:
                    payload = json.loads(msg)
                    cmd = payload.get("command")
                    if cmd == "subscribe_eeg":
                        async with self._lock:
                            if ws not in self._eeg_clients:
                                self._eeg_clients.append(ws)
                    elif cmd == "unsubscribe_eeg":
                        async with self._lock:
                            if ws in self._eeg_clients:
                                self._eeg_clients.remove(ws)
                except json.JSONDecodeError:
                    pass
        except WebSocketDisconnect:
            pass
        finally:
            await self.disconnect(ws)


ws_manager = WebSocketManager()
