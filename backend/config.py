import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# ── EEG Hardware ──────────────────────────────────────────────
EEG_SAMPLE_RATE = 256  # Muse 2 sampling rate (Hz)
EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
NUM_CHANNELS = len(EEG_CHANNELS)

# ── Signal Processing ────────────────────────────────────────
BANDPASS_LOW = 1.0   # Hz
BANDPASS_HIGH = 30.0  # Hz
BANDPASS_ORDER = 4    # Butterworth filter order
EPOCH_TMIN = -0.1     # seconds before stimulus onset
EPOCH_TMAX = 0.8      # seconds after stimulus onset
EPOCH_SAMPLES = int((EPOCH_TMAX - EPOCH_TMIN) * EEG_SAMPLE_RATE)  # ~230

# ── P300 Stimulus ────────────────────────────────────────────
FLASH_DURATION_MS = 150   # highlight duration per phrase
ISI_MS = 100              # inter-stimulus interval
NUM_FLASH_ROUNDS = 5      # rounds before classification decision
NUM_PHRASES = 6           # phrases displayed at once

# ── Artifact Detection ───────────────────────────────────────
BLINK_THRESHOLD_UV = 150      # peak amplitude on AF7/AF8
BLINK_REFRACTORY_MS = 500     # minimum gap between blink events
CLENCH_RMS_THRESHOLD = 50     # RMS threshold on TP9/TP10
CLENCH_MIN_DURATION_MS = 300  # sustained above threshold

# ── Classifier ───────────────────────────────────────────────
MODEL_PATH = str(BASE_DIR / "models" / "p300_lda.joblib")
CALIBRATION_TARGET_EPOCHS = 60  # target + non-target combined
DOWNSAMPLE_FACTOR = 8  # epoch downsampling for feature extraction

# ── Redis ─────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_RAW_STREAM = "eeg:raw"
REDIS_EPOCH_STREAM = "eeg:epochs"
REDIS_EVENT_STREAM = "eeg:events"
REDIS_RAW_MAXLEN = 10000  # ~40 seconds of raw data

# ── Gemini LLM ────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"

# ── FastAPI ───────────────────────────────────────────────────
FASTAPI_HOST = "0.0.0.0"
FASTAPI_PORT = 8000

# ── Simulation Mode ──────────────────────────────────────────
SIMULATE_EEG = os.getenv("SIMULATE_EEG", "false").lower() == "true"
