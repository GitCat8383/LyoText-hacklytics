"""REST API routes for the React frontend."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import config
from database.store import redis_store
from eeg.classifier import p300_classifier
from eeg.processing import signal_processor
from llm.phrase_engine import phrase_engine
from utils.events import Event, EventType, event_bus

router = APIRouter(prefix="/api")


class ConfigUpdate(BaseModel):
    blink_threshold_uv: float | None = None
    clench_rms_threshold: float | None = None
    clench_min_duration_ms: float | None = None
    flash_duration_ms: int | None = None
    isi_ms: int | None = None
    num_flash_rounds: int | None = None


class StatusResponse(BaseModel):
    eeg_connected: bool
    classifier_loaded: bool
    classifier_calibrating: bool
    calibration_epochs: int
    redis_connected: bool
    simulate_mode: bool


@router.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    from eeg.stream import eeg_stream
    return StatusResponse(
        eeg_connected=eeg_stream.is_running,
        classifier_loaded=p300_classifier.is_trained,
        classifier_calibrating=p300_classifier.is_calibrating,
        calibration_epochs=p300_classifier.calibration_count,
        redis_connected=redis_store.ping(),
        simulate_mode=config.SIMULATE_EEG,
    )


@router.get("/phrases")
async def get_phrases() -> dict[str, Any]:
    phrases = await phrase_engine.generate_phrases()
    return {"phrases": phrases}


@router.post("/phrases/confirm/{index}")
async def confirm_phrase(index: int) -> dict[str, Any]:
    phrases = await phrase_engine.generate_phrases()
    if index < 0 or index >= len(phrases):
        raise HTTPException(status_code=400, detail="Invalid phrase index")
    phrase = phrases[index]
    phrase_engine.confirm_phrase(phrase)

    event_bus.emit(Event(
        type=EventType.PHRASE_CONFIRMED,
        data={"phrase": phrase, "history": phrase_engine.history},
    ))

    new_phrases = await phrase_engine.generate_phrases()
    event_bus.emit(Event(
        type=EventType.PHRASES_UPDATED,
        data={"phrases": new_phrases},
    ))

    return {"confirmed": phrase, "history": phrase_engine.history, "new_phrases": new_phrases}


@router.get("/history")
async def get_history() -> dict[str, Any]:
    return {"history": phrase_engine.history}


@router.delete("/history/last")
async def delete_last() -> dict[str, Any]:
    removed = phrase_engine.delete_last()
    if removed is None:
        raise HTTPException(status_code=404, detail="No history to delete")

    event_bus.emit(Event(
        type=EventType.PHRASE_DELETED,
        data={"removed": removed, "history": phrase_engine.history},
    ))
    return {"removed": removed, "history": phrase_engine.history}


@router.post("/calibration/start")
async def start_calibration() -> dict[str, str]:
    if p300_classifier.is_calibrating:
        raise HTTPException(status_code=409, detail="Calibration already in progress")
    p300_classifier.start_calibration()
    return {"status": "calibration_started"}


@router.post("/calibration/stop")
async def stop_calibration() -> dict[str, Any]:
    if not p300_classifier.is_calibrating:
        raise HTTPException(status_code=409, detail="No calibration in progress")
    try:
        accuracy = p300_classifier.finish_calibration()
        return {"status": "calibration_complete", "accuracy": accuracy}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/config")
async def get_config() -> dict[str, Any]:
    return {
        "eeg_sample_rate": config.EEG_SAMPLE_RATE,
        "bandpass_low": config.BANDPASS_LOW,
        "bandpass_high": config.BANDPASS_HIGH,
        "epoch_tmin": config.EPOCH_TMIN,
        "epoch_tmax": config.EPOCH_TMAX,
        "flash_duration_ms": config.FLASH_DURATION_MS,
        "isi_ms": config.ISI_MS,
        "num_flash_rounds": config.NUM_FLASH_ROUNDS,
        "num_phrases": config.NUM_PHRASES,
        "blink_threshold_uv": config.BLINK_THRESHOLD_UV,
        "clench_rms_threshold": config.CLENCH_RMS_THRESHOLD,
        "clench_min_duration_ms": config.CLENCH_MIN_DURATION_MS,
    }


@router.patch("/config")
async def update_config(update: ConfigUpdate) -> dict[str, str]:
    if update.blink_threshold_uv is not None:
        config.BLINK_THRESHOLD_UV = update.blink_threshold_uv
    if update.clench_rms_threshold is not None:
        config.CLENCH_RMS_THRESHOLD = update.clench_rms_threshold
    if update.clench_min_duration_ms is not None:
        config.CLENCH_MIN_DURATION_MS = update.clench_min_duration_ms
    if update.flash_duration_ms is not None:
        config.FLASH_DURATION_MS = update.flash_duration_ms
    if update.isi_ms is not None:
        config.ISI_MS = update.isi_ms
    if update.num_flash_rounds is not None:
        config.NUM_FLASH_ROUNDS = update.num_flash_rounds
    return {"status": "config_updated"}


@router.get("/events")
async def get_events(seconds: float = 60.0) -> dict[str, Any]:
    events = redis_store.get_recent_events(seconds)
    return {"events": events}


@router.get("/eeg/band_power")
async def get_band_power() -> dict[str, Any]:
    result = signal_processor.compute_band_power()
    return result


@router.get("/eeg/raw/{offset_sec}")
async def get_raw_at_second(offset_sec: float) -> dict[str, Any]:
    """Read ~1 second of raw EEG data from `offset_sec` seconds ago.

    Example: GET /api/eeg/raw/10 → returns ~256 samples from 10 seconds ago.
    """
    if offset_sec < 0:
        raise HTTPException(status_code=400, detail="offset_sec must be >= 0")
    max_seconds = config.REDIS_RAW_MAXLEN / config.EEG_SAMPLE_RATE
    if offset_sec > max_seconds:
        raise HTTPException(
            status_code=400,
            detail=f"offset_sec too large. Redis keeps ~{max_seconds:.0f}s of data.",
        )
    return redis_store.get_raw_at_second(offset_sec)
