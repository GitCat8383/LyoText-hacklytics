"""REST API routes for the React frontend."""

from __future__ import annotations

import threading
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

import config
from database.store import redis_store
from eeg.classifier import p300_classifier
from eeg.collector import data_collector
from eeg.dataset import list_sessions, load_all_sessions, load_epochs
from eeg.deep_trainer import deep_trainer
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
    eegnet_p300: bool
    eegnet_gesture: bool
    dl_training: bool
    collecting_data: bool


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
        eegnet_p300=deep_trainer.p300_model is not None,
        eegnet_gesture=deep_trainer.gesture_model is not None,
        dl_training=deep_trainer.is_training,
        collecting_data=data_collector.is_collecting,
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


# ── Deep Learning: Data Collection ────────────────────────────


class CollectionRequest(BaseModel):
    name: str
    gesture_types: list[str] | None = None
    trials_per_gesture: int = 30


class ManualEpochRequest(BaseModel):
    label: str


class TrainRequest(BaseModel):
    model_type: str = "gesture"  # "p300" | "gesture"
    session_name: str | None = None
    max_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    patience: int = 15


@router.post("/dl/collect/start")
async def start_collection(req: CollectionRequest) -> dict[str, Any]:
    """Start a guided data collection session."""
    try:
        status = data_collector.start_session(
            name=req.name,
            gesture_types=req.gesture_types,
            trials_per_gesture=req.trials_per_gesture,
        )
        return {"status": "started", **status}
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/dl/collect/stop")
async def stop_collection() -> dict[str, Any]:
    """Stop the current collection session and save data."""
    try:
        result = data_collector.stop_session()
        return {"status": "stopped", **result}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/dl/collect/pause")
async def pause_collection() -> dict[str, str]:
    data_collector.pause_session()
    return {"status": "paused"}


@router.post("/dl/collect/resume")
async def resume_collection() -> dict[str, str]:
    data_collector.resume_session()
    return {"status": "resumed"}


@router.get("/dl/collect/status")
async def collection_status() -> dict[str, Any]:
    status = data_collector.session_status
    if status is None:
        return {"active": False}
    return {"active": True, **status}


@router.post("/dl/collect/manual")
async def add_manual_epoch(req: ManualEpochRequest) -> dict[str, Any]:
    """Manually label the current 1-second EEG window."""
    try:
        result = data_collector.add_manual_epoch(req.label)
        return result
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/dl/collect/save")
async def save_manual_epochs(session_name: str | None = Query(None)) -> dict[str, Any]:
    """Save manually collected epochs to disk."""
    try:
        result = data_collector.save_manual(session_name)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Deep Learning: Data Sessions ──────────────────────────────

@router.get("/dl/sessions")
async def get_sessions() -> dict[str, Any]:
    """List all saved data collection sessions."""
    return {"sessions": list_sessions()}


@router.get("/dl/sessions/{name}")
async def get_session_detail(name: str) -> dict[str, Any]:
    """Get details of a specific session."""
    try:
        epochs, labels = load_epochs(name)
        from eeg.dataset import LABEL_NAMES
        class_dist = {}
        for idx in set(labels.tolist()):
            lbl = LABEL_NAMES[idx] if idx < len(LABEL_NAMES) else str(idx)
            class_dist[lbl] = int((labels == idx).sum())

        return {
            "name": name,
            "n_epochs": len(labels),
            "shape": list(epochs.shape),
            "class_distribution": class_dist,
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session '{name}' not found")


# ── Deep Learning: Training ──────────────────────────────────

@router.post("/dl/train")
async def start_training(req: TrainRequest) -> dict[str, Any]:
    """Train an EEGNet model on collected data (runs in background thread)."""
    if deep_trainer.is_training:
        raise HTTPException(status_code=409, detail="Training already in progress")

    # Load data
    try:
        if req.session_name:
            epochs, labels = load_epochs(req.session_name)
        else:
            epochs, labels = load_all_sessions()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")

    if len(labels) < 20:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 20 epochs for training, got {len(labels)}",
        )

    def _train_bg():
        try:
            if req.model_type == "p300":
                deep_trainer.train_p300(
                    epochs, labels,
                    max_epochs=req.max_epochs,
                    batch_size=req.batch_size,
                    lr=req.learning_rate,
                    patience=req.patience,
                )
            else:
                deep_trainer.train_gesture(
                    epochs, labels,
                    max_epochs=req.max_epochs,
                    batch_size=req.batch_size,
                    lr=req.learning_rate,
                    patience=req.patience,
                )
        except Exception:
            import logging
            logging.getLogger(__name__).exception("Background training failed")

    threading.Thread(target=_train_bg, daemon=True).start()

    return {
        "status": "training_started",
        "model_type": req.model_type,
        "n_epochs": len(labels),
        "n_classes": len(set(labels.tolist())),
    }


@router.get("/dl/train/status")
async def training_status() -> dict[str, Any]:
    """Check if training is in progress."""
    return {
        "training": deep_trainer.is_training,
        "p300_loaded": deep_trainer.p300_model is not None,
        "gesture_loaded": deep_trainer.gesture_model is not None,
    }


@router.get("/dl/models")
async def get_models() -> dict[str, Any]:
    """List available trained models."""
    from pathlib import Path
    models_dir = Path(config.BASE_DIR) / "models"
    models = []
    if models_dir.exists():
        for f in models_dir.glob("eegnet_*.pt"):
            import torch
            try:
                state = torch.load(f, map_location="cpu", weights_only=True)
                models.append({
                    "name": f.stem,
                    "file": f.name,
                    "n_channels": state.get("n_channels"),
                    "n_samples": state.get("n_samples"),
                    "n_classes": state.get("n_classes"),
                    "size_kb": f.stat().st_size // 1024,
                })
            except Exception:
                models.append({"name": f.stem, "file": f.name, "error": "failed to load"})

    return {"models": models}


@router.post("/dl/models/reload")
async def reload_models() -> dict[str, Any]:
    """Reload EEGNet models from disk."""
    result = deep_trainer.load_models()
    return {"status": "reloaded", "loaded": result}


@router.post("/dl/predict/gesture")
async def predict_gesture_now() -> dict[str, Any]:
    """Run gesture prediction on the current 1-second raw EEG window."""
    if deep_trainer.gesture_model is None:
        raise HTTPException(status_code=400, detail="No gesture model loaded")

    try:
        import numpy as _np
        raw_samples = redis_store.get_recent_raw(seconds=1.0)
        if len(raw_samples) < config.EEG_SAMPLE_RATE * 0.8:
            raise HTTPException(status_code=400, detail="Not enough EEG data yet")

        data = _np.array([
            [s["tp9"], s["af7"], s["af8"], s["tp10"]]
            for s in raw_samples
        ], dtype=_np.float32)
        target = config.EEG_SAMPLE_RATE
        if len(data) > target:
            data = data[-target:]
        elif len(data) < target:
            data = _np.pad(data, ((target - len(data), 0), (0, 0)), mode="edge")
        window = data.T

        cls_idx, cls_name, confidence = deep_trainer.predict_gesture(window)
        return {
            "class_index": cls_idx,
            "class_name": cls_name,
            "confidence": round(confidence, 4),
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Live Test Mode ─────────────────────────────────────────

@router.post("/dl/live-test/start")
async def start_live_test(request: Request) -> dict[str, Any]:
    """Enable live test mode — streams gesture predictions via WebSocket."""
    orch = request.app.state.orchestrator
    if deep_trainer.gesture_model is None:
        raise HTTPException(status_code=400, detail="No gesture model loaded")
    orch.live_test_active = True
    return {"status": "started", "message": "Live test mode enabled"}


@router.post("/dl/live-test/stop")
async def stop_live_test(request: Request) -> dict[str, Any]:
    """Disable live test mode."""
    orch = request.app.state.orchestrator
    orch.live_test_active = False
    return {"status": "stopped", "message": "Live test mode disabled"}


@router.get("/dl/live-test/status")
async def live_test_status(request: Request) -> dict[str, Any]:
    """Check if live test mode is active."""
    orch = request.app.state.orchestrator
    return {
        "active": orch.live_test_active,
        "model_loaded": deep_trainer.gesture_model is not None,
    }
