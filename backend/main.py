"""Hacklytic BCI — Main entry point.

Launches:
1. FastAPI server (with WebSocket) on the main thread via Uvicorn
2. Muse 2 EEG streaming (background thread)
3. Signal processing pipeline (background thread)
4. Pygame stimulus window (separate process)
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import sys
import threading
import time

import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from api.server import create_app
from database.store import redis_store
from eeg.artifacts import artifact_detector
from eeg.classifier import p300_classifier
from eeg.processing import signal_processor
from eeg.stream import eeg_stream
from llm.phrase_engine import phrase_engine
from stimulus.flasher import StimulusFlasher
from utils.events import Event, EventType, event_bus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("hacklytic")


class BCIOrchestrator:

    def __init__(self) -> None:
        self._sample_queue: queue.Queue = queue.Queue(maxsize=1024)
        self._flasher = StimulusFlasher()
        self._running = False
        self._processing_thread: threading.Thread | None = None
        self._stimulus_thread: threading.Thread | None = None
        self._current_phrases: list[str] = []
        self._cycle_epochs: list[tuple] = []
        self._flash_active = False
        self._last_p300_selection: tuple[int, str] | None = None

    def start(self) -> None:
        self._running = True

        if p300_classifier.load():
            logger.info("P300 model loaded from disk — skipping calibration")
        else:
            logger.info("No saved model found — calibration will be required")

        eeg_stream.start(sample_buffer=self._sample_queue)
        logger.info("EEG stream started (simulate=%s)", config.SIMULATE_EEG)

        self._flasher.start()
        logger.info("Pygame stimulus window launched")

        self._processing_thread = threading.Thread(
            target=self._processing_loop, daemon=True
        )
        self._processing_thread.start()

        self._stimulus_thread = threading.Thread(
            target=self._stimulus_event_loop, daemon=True
        )
        self._stimulus_thread.start()

        event_bus.on(EventType.BLINK_DETECTED, self._on_blink)
        event_bus.on(EventType.CLENCH_DETECTED, self._on_clench)

        threading.Thread(target=self._async_init, daemon=True).start()

    def _async_init(self) -> None:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._load_initial_phrases())
        except Exception:
            logger.exception("Async init failed")
        finally:
            loop.close()

    async def _load_initial_phrases(self) -> None:
        try:
            self._current_phrases = await phrase_engine.generate_phrases()
            self._flasher.set_phrases(self._current_phrases)
            event_bus.emit(Event(
                type=EventType.PHRASES_UPDATED,
                data={"phrases": self._current_phrases},
            ))
            logger.info("Initial phrases loaded: %s", self._current_phrases)

            if p300_classifier.is_trained:
                time.sleep(1.0)
                self._start_flash_cycle()
        except Exception:
            logger.exception("Failed to load initial phrases")

    def stop(self) -> None:
        self._running = False
        eeg_stream.stop()
        self._flasher.stop()
        if self._processing_thread:
            self._processing_thread.join(timeout=3)
        if self._stimulus_thread:
            self._stimulus_thread.join(timeout=3)

    def _processing_loop(self) -> None:
        while self._running:
            try:
                samples, timestamps = self._sample_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            filtered = signal_processor.process_chunk(samples, timestamps)
            artifact_detector.process_samples(filtered, timestamps)

            epochs = signal_processor.try_extract_epochs()
            for epoch, phrase_idx in epochs:
                if not p300_classifier.is_calibrating:
                    self._cycle_epochs.append((epoch, phrase_idx))

    def _stimulus_event_loop(self) -> None:
        while self._running:
            try:
                ev = self._flasher.event_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            ev_type = ev.get("event")

            if ev_type == "stimulus_onset":
                event_bus.emit(Event(
                    type=EventType.STIMULUS_ONSET,
                    data={"phrase_index": ev["phrase_index"]},
                    timestamp=ev["timestamp"],
                ))
            elif ev_type == "flash_cycle_complete":
                self._flash_active = False
                artifact_detector.set_flash_active(False)
                self._on_flash_cycle_complete()

    def _start_flash_cycle(self) -> None:
        self._cycle_epochs.clear()
        self._flash_active = True
        self._last_p300_selection = None
        artifact_detector.set_flash_active(True)
        self._flasher.start_flash()
        logger.info("Flash cycle started")

    def _on_flash_cycle_complete(self) -> None:
        if not p300_classifier.is_trained:
            logger.warning("Flash complete but classifier not trained")
            return

        if not self._cycle_epochs:
            logger.warning("No epochs collected during flash cycle")
            self._schedule_next_cycle(delay=1.0)
            return

        try:
            winner_idx, confidence = p300_classifier.select_phrase(
                self._cycle_epochs, config.NUM_PHRASES
            )
            winner_phrase = (
                self._current_phrases[winner_idx]
                if winner_idx < len(self._current_phrases)
                else "?"
            )

            self._last_p300_selection = (winner_idx, winner_phrase)
            logger.info(
                "P300 result: phrase[%d]='%s' (confidence=%.2f)",
                winner_idx, winner_phrase, confidence,
            )

            self._flasher.highlight(winner_idx, "green")

            event_bus.emit(Event(
                type=EventType.P300_RESULT,
                data={
                    "selected_index": winner_idx,
                    "confidence": confidence,
                    "phrase": winner_phrase,
                },
            ))

            try:
                redis_store.push_event("p300_result", {
                    "selected_index": winner_idx,
                    "confidence": confidence,
                    "phrase": winner_phrase,
                })
            except Exception:
                logger.debug("Redis push failed (non-critical)")

        except Exception:
            logger.exception("Classification failed")
            self._schedule_next_cycle(delay=1.0)

    def _on_blink(self, event: Event) -> None:
        if self._flash_active:
            return
        if self._last_p300_selection is None:
            return

        idx, phrase = self._last_p300_selection
        self._last_p300_selection = None

        phrase_engine.confirm_phrase(phrase)
        logger.info("Phrase confirmed by blink: '%s'", phrase)

        event_bus.emit(Event(
            type=EventType.PHRASE_CONFIRMED,
            data={"phrase": phrase, "history": phrase_engine.history},
        ))

        try:
            redis_store.push_event("phrase_confirmed", {"phrase": phrase})
        except Exception:
            pass

        self._flasher.reset_highlight()
        self._refresh_phrases_and_flash()

    def _on_clench(self, event: Event) -> None:
        if self._flash_active:
            return

        removed = phrase_engine.delete_last()
        if removed:
            logger.info("Phrase deleted by jaw clench: '%s'", removed)
            self._last_p300_selection = None
            event_bus.emit(Event(
                type=EventType.PHRASE_DELETED,
                data={"removed": removed, "history": phrase_engine.history},
            ))
            try:
                redis_store.push_event("phrase_deleted", {"removed": removed})
            except Exception:
                pass
            self._flasher.reset_highlight()
            self._refresh_phrases_and_flash()
        else:
            self._last_p300_selection = None
            self._flasher.reset_highlight()
            self._schedule_next_cycle(delay=0.5)

    def _refresh_phrases_and_flash(self) -> None:
        def _do():
            loop = asyncio.new_event_loop()
            try:
                phrases = loop.run_until_complete(phrase_engine.generate_phrases())
                self._current_phrases = phrases
                self._flasher.set_phrases(phrases)
                event_bus.emit(Event(
                    type=EventType.PHRASES_UPDATED,
                    data={"phrases": phrases},
                ))
                time.sleep(0.5)
                self._start_flash_cycle()
            except Exception:
                logger.exception("Failed to refresh phrases")
            finally:
                loop.close()

        threading.Thread(target=_do, daemon=True).start()

    def _schedule_next_cycle(self, delay: float = 0.5) -> None:
        def _delayed():
            time.sleep(delay)
            if self._running:
                self._start_flash_cycle()
        threading.Thread(target=_delayed, daemon=True).start()


def main() -> None:
    app = create_app()
    app.state.orchestrator = BCIOrchestrator()

    logger.info(
        "Starting Hacklytic BCI on %s:%d (simulate=%s)",
        config.FASTAPI_HOST,
        config.FASTAPI_PORT,
        config.SIMULATE_EEG,
    )

    uvicorn.run(
        app,
        host=config.FASTAPI_HOST,
        port=config.FASTAPI_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
