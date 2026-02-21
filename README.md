# Hacklytic — Hybrid BCI Assistive Communication

A brain-computer interface that uses **Muse 2** consumer EEG hardware to detect **P300 attention-evoked potentials**, combined with **Gemini-powered phrase prediction**, to enable rapid assistive communication.

## How It Works

1. **Muse 2** streams 4-channel EEG (TP9, AF7, AF8, TP10) via Lab Streaming Layer
2. A real-time signal pipeline applies bandpass filtering and stimulus-locked epoching
3. An **LDA classifier** detects P300 responses to identify which phrase the user is focusing on
4. **Gemini Flash** generates 6 contextual phrase suggestions based on conversation history
5. A **Pygame window** flashes phrases in sequence (oddball paradigm)
6. **Blink** to confirm a selected phrase, **jaw clench** to delete/undo

## Architecture

```
Muse 2 → muselsl → LSL → FastAPI Backend → Pygame Stimulus Window
                              ↕ WebSocket        ↕ IPC
                         React Frontend    Flash + Markers
```

- **FastAPI backend** — central hub (EEG pipeline, Redis, LDA classifier, Gemini API)
- **Pygame window** — precise stimulus flashing for P300 oddball paradigm
- **React frontend** — dashboard and controls (candy-sky themed)

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Redis server
- Muse 2 headband (or use simulated mode)
- Gemini API key

### Backend Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your GEMINI_API_KEY
```

### Frontend Setup

```bash
cd frontend
npm install

cp .env.example .env
# Edit .env with your GEMINI_API_KEY (for TTS)
```

### Run

**Option A — Simulated EEG (no Muse 2 needed):**

```bash
# Terminal 1: Start Redis
redis-server

# Terminal 2: Start backend (simulated EEG)
cd backend
source .venv/bin/activate
SIMULATE_EEG=true python main.py

# Terminal 3: Start frontend
cd frontend
npm run dev
```

- Backend API: `http://localhost:8000` (Swagger docs at `/docs`)
- Frontend: `http://localhost:3000`
- Pygame stimulus window opens automatically

### First-Time Calibration

If no saved model exists, click "Calibrate" in the frontend. Follow the Pygame prompts (~2-3 minutes). The model saves to `backend/models/p300_lda.joblib` and reloads automatically on restart.

## Tech Stack

**Backend:** FastAPI, muselsl, pylsl, Redis, SciPy, scikit-learn, Pygame, Gemini API
**Frontend:** React 19, Vite, TypeScript, Tailwind CSS, Recharts, Framer Motion, Gemini TTS
