import React, { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { speakText } from '../services/gemini';
import {
  bciSocket,
  getPhrases,
  getStatus,
  getHistory,
  confirmPhrase,
  deleteLastPhrase,
  startCalibration,
  stopCalibration,
  type BCIEvent,
  type SystemStatus,
} from '../services/api';
import {
  Volume2,
  BrainCircuit,
  Play,
  StopCircle,
  RefreshCw,
  Trash2,
  Wifi,
  WifiOff,
  Eye,
  Zap,
  Settings,
} from 'lucide-react';
import EEGMonitor from './EEGMonitor';
import BandPowerHistogram from './BandPowerHistogram';

const FALLBACK_PHRASES = ["Yes", "No", "Help me please", "Thank you", "I need water", "I'm okay"];

const P300Grid: React.FC = () => {
  const [phrases, setPhrases] = useState<string[]>(FALLBACK_PHRASES);
  const [sentence, setSentence] = useState<string[]>([]);
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [isFlashing, setIsFlashing] = useState(false);
  const [flashIndex, setFlashIndex] = useState<number | null>(null);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [lastEvent, setLastEvent] = useState<string>('');
  const [isDemoMode, setIsDemoMode] = useState(false);
  const [confidence, setConfidence] = useState<number | null>(null);

  // Blink / clench visual feedback
  const [blinkFlash, setBlinkFlash] = useState(false);
  const [clenchFlash, setClenchFlash] = useState(false);

  const sentenceRef = useRef(sentence);
  useEffect(() => { sentenceRef.current = sentence; }, [sentence]);
  const isDemoModeRef = useRef(isDemoMode);
  useEffect(() => { isDemoModeRef.current = isDemoMode; }, [isDemoMode]);

  // ── Connect to backend on mount ────────────────────────────

  useEffect(() => {
    bciSocket.connect();

    const fetchInitial = async () => {
      try {
        const [st, ph, hist] = await Promise.all([
          getStatus(),
          getPhrases(),
          getHistory(),
        ]);
        setStatus(st);
        setPhrases(ph.length > 0 ? ph : FALLBACK_PHRASES);
        setSentence(hist);
      } catch {
        console.warn('Backend not reachable, using fallback mode');
      }
    };
    fetchInitial();

    return () => bciSocket.disconnect();
  }, []);

  // ── WebSocket event handlers ───────────────────────────────

  useEffect(() => {
    const unsubs: (() => void)[] = [];

    unsubs.push(
      bciSocket.on('ws_connected', () => setWsConnected(true)),
      bciSocket.on('ws_disconnected', () => setWsConnected(false)),

      bciSocket.on('phrases_updated', (e: BCIEvent) => {
        const newPhrases = e.data.phrases;
        if (Array.isArray(newPhrases) && newPhrases.length > 0) {
          setPhrases(newPhrases);
        }
      }),

      bciSocket.on('p300_result', (e: BCIEvent) => {
        const idx = e.data.selected_index;
        const conf = e.data.confidence;
        setSelectedIndex(idx);
        setConfidence(conf);
        setIsFlashing(false);
        setFlashIndex(null);
        setLastEvent(`P300 → "${e.data.phrase}" (${(conf * 100).toFixed(0)}%)`);
      }),

      bciSocket.on('blink_detected', () => {
        setBlinkFlash(true);
        setTimeout(() => setBlinkFlash(false), 500);
        setLastEvent('Blink detected');
      }),

      bciSocket.on('clench_detected', () => {
        setClenchFlash(true);
        setTimeout(() => setClenchFlash(false), 500);
        setLastEvent('Jaw clench detected');
      }),

      bciSocket.on('phrase_confirmed', (e: BCIEvent) => {
        setSentence(e.data.history || []);
        setSelectedIndex(null);
        setConfidence(null);
        setLastEvent(`Confirmed: "${e.data.phrase}"`);
      }),

      bciSocket.on('phrase_deleted', (e: BCIEvent) => {
        setSentence(e.data.history || []);
        setSelectedIndex(null);
        setLastEvent(`Deleted: "${e.data.removed}"`);
      }),

      bciSocket.on('calibration_progress', (e: BCIEvent) => {
        setLastEvent(`Calibration: ${e.data.epochs_collected}/${e.data.target} epochs`);
      }),

      bciSocket.on('system_status', (e: BCIEvent) => {
        setStatus(e.data as any);
      }),
    );

    return () => unsubs.forEach((u) => u());
  }, []);

  // ── Simulated flashing (demo mode visual only) ─────────────

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;

    if (isFlashing) {
      interval = setInterval(() => {
        const idx = Math.floor(Math.random() * phrases.length);
        setFlashIndex(idx);
        setTimeout(() => setFlashIndex(null), 100);
      }, 175);
    } else {
      setFlashIndex(null);
    }
    return () => clearInterval(interval);
  }, [isFlashing, phrases.length]);

  // Demo mode auto-selection
  useEffect(() => {
    if (!isDemoMode || !isFlashing) return;
    const timeout = setTimeout(() => {
      const idx = Math.floor(Math.random() * phrases.length);
      setSelectedIndex(idx);
      setConfidence(0.7 + Math.random() * 0.25);
      setIsFlashing(false);
      setLastEvent(`P300 → "${phrases[idx]}" (demo)`);
    }, 3000 + Math.random() * 2000);
    return () => clearTimeout(timeout);
  }, [isDemoMode, isFlashing, phrases]);

  // ── Actions ────────────────────────────────────────────────

  const handlePhraseClick = useCallback(async (index: number) => {
    setIsLoading(true);
    try {
      const result = await confirmPhrase(index);
      setSentence(result.history);
      setPhrases(result.new_phrases.length > 0 ? result.new_phrases : FALLBACK_PHRASES);
      setSelectedIndex(null);
      setConfidence(null);
    } catch {
      // Fallback: local selection
      const phrase = phrases[index];
      setSentence(prev => [...prev, phrase]);
      setSelectedIndex(null);
    }
    setIsLoading(false);
  }, [phrases]);

  const handleDemoConfirm = useCallback(async () => {
    if (selectedIndex === null) return;
    await handlePhraseClick(selectedIndex);
    setTimeout(() => setIsFlashing(true), 500);
  }, [selectedIndex, handlePhraseClick]);

  const toggleSession = () => {
    const next = !isSessionActive;
    setIsSessionActive(next);
    setIsFlashing(next);
    setSelectedIndex(null);
    setConfidence(null);
  };

  const handleSpeak = async () => {
    if (sentence.length === 0) return;
    const text = sentence.join(" ");
    setIsLoading(true);
    const wasFlashing = isFlashing;
    if (wasFlashing) setIsFlashing(false);

    const audioDataUrl = await speakText(text);
    if (audioDataUrl) {
      const audio = new Audio(audioDataUrl);
      audio.onended = () => { if (wasFlashing) setIsFlashing(true); };
      audio.play();
    } else {
      if (wasFlashing) setIsFlashing(true);
    }
    setIsLoading(false);
  };

  const handleDelete = async () => {
    try {
      const result = await deleteLastPhrase();
      setSentence(result.history);
    } catch {
      setSentence(prev => prev.slice(0, -1));
    }
  };

  const clearSentence = () => {
    setSentence([]);
    setSelectedIndex(null);
    setIsFlashing(false);
    setIsSessionActive(false);
  };

  const handleCalibrate = async () => {
    try {
      if (status?.classifier_calibrating) {
        const result = await stopCalibration();
        setLastEvent(`Calibration done! Accuracy: ${(result.accuracy * 100).toFixed(1)}%`);
      } else {
        await startCalibration();
        setLastEvent('Calibration started — follow Pygame prompts');
      }
      const st = await getStatus();
      setStatus(st);
    } catch (e: any) {
      setLastEvent(`Calibration error: ${e.message}`);
    }
  };

  return (
    <div className="flex flex-col h-full max-w-3xl mx-auto p-4 gap-4">
      {/* Status Bar */}
      <div className="flex items-center justify-between bg-white/30 backdrop-blur-md rounded-full px-4 py-2 border border-white/40">
        <div className="flex items-center gap-3">
          <div className={`flex items-center gap-1.5 text-xs font-bold ${wsConnected ? 'text-emerald-600' : 'text-red-500'}`}>
            {wsConnected ? <Wifi size={14} /> : <WifiOff size={14} />}
            {wsConnected ? 'Connected' : 'Offline'}
          </div>
          {status?.simulate_mode && (
            <span className="text-[10px] font-bold bg-amber-200 text-amber-800 px-2 py-0.5 rounded-full">SIM</span>
          )}
          {status?.classifier_loaded && (
            <span className="text-[10px] font-bold bg-emerald-200 text-emerald-800 px-2 py-0.5 rounded-full">MODEL OK</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {/* Blink indicator */}
          <div className={`flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold transition-all duration-200 ${blinkFlash ? 'bg-blue-400 text-white scale-110' : 'bg-blue-100 text-blue-500'}`}>
            <Eye size={10} /> BLINK
          </div>
          {/* Clench indicator */}
          <div className={`flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold transition-all duration-200 ${clenchFlash ? 'bg-orange-400 text-white scale-110' : 'bg-orange-100 text-orange-500'}`}>
            <Zap size={10} /> CLENCH
          </div>
        </div>
      </div>

      {/* Start/Stop + Calibrate */}
      <div className="flex justify-center gap-3 py-1">
        <button
          onClick={toggleSession}
          className={`
            relative group overflow-hidden px-8 py-3 rounded-full font-black text-lg text-white shadow-xl transition-all transform hover:scale-105 active:scale-95
            ${isSessionActive
              ? 'bg-gradient-to-b from-red-400 to-red-600 shadow-red-500/40 border-2 border-red-300'
              : 'bg-gradient-to-b from-pink-400 to-purple-500 shadow-purple-500/40 border-2 border-pink-300'
            }
          `}
        >
          <div className="absolute inset-0 bg-white/20 group-hover:bg-white/30 transition-colors" />
          <div className="absolute top-0 left-0 w-full h-1/2 bg-white/20 rounded-t-full blur-[1px]" />
          <div className="flex items-center gap-3 relative z-10 drop-shadow-md">
            {isSessionActive ? <StopCircle size={24} fill="currentColor" /> : <Play size={24} fill="currentColor" />}
            <span className="tracking-wide">{isSessionActive ? 'STOP' : 'START'}</span>
          </div>
        </button>

        <button
          onClick={handleCalibrate}
          className={`
            relative group overflow-hidden px-5 py-3 rounded-full font-bold text-sm text-white shadow-lg transition-all transform hover:scale-105 active:scale-95
            ${status?.classifier_calibrating
              ? 'bg-gradient-to-b from-amber-400 to-amber-600 border-2 border-amber-300'
              : 'bg-gradient-to-b from-cyan-400 to-cyan-600 border-2 border-cyan-300'
            }
          `}
        >
          <div className="absolute inset-0 bg-white/20 group-hover:bg-white/30 transition-colors" />
          <div className="flex items-center gap-2 relative z-10 drop-shadow-md">
            <BrainCircuit size={18} />
            <span>{status?.classifier_calibrating ? 'STOP CAL' : 'CALIBRATE'}</span>
          </div>
        </button>
      </div>

      {/* EEG Monitor */}
      <div className="bg-white/40 backdrop-blur-md p-2 rounded-2xl border border-white/50 shadow-sm">
        <div className="bg-white/60 rounded-xl p-1">
          <EEGMonitor isFlashing={isFlashing} />
        </div>
      </div>

      {/* Band Power Histogram */}
      <div className="bg-white/40 backdrop-blur-md p-2 rounded-2xl border border-white/50 shadow-sm">
        <div className="bg-white/60 rounded-xl p-1">
          <BandPowerHistogram />
        </div>
      </div>

      {/* Control Panel */}
      <div className="bg-white/40 backdrop-blur-md p-4 rounded-3xl border border-white/50 shadow-sm space-y-3">
        <div className="flex items-center justify-between px-2 pb-2">
          <span className="text-xs font-bold text-zinc-500 uppercase tracking-wider">Controls</span>
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold text-blue-500">Demo Mode</span>
            <button
              onClick={() => setIsDemoMode(!isDemoMode)}
              className={`w-8 h-4 rounded-full relative transition-colors duration-300 ${isDemoMode ? 'bg-blue-400' : 'bg-zinc-300'}`}
            >
              <div className={`absolute top-0.5 left-0.5 w-3 h-3 bg-white rounded-full shadow-sm transition-transform duration-300 ${isDemoMode ? 'translate-x-4' : 'translate-x-0'}`} />
            </button>
          </div>
        </div>

        <button
          onClick={handleSpeak}
          disabled={sentence.length === 0 || isLoading}
          className="w-full group relative overflow-hidden bg-gradient-to-r from-purple-400 to-purple-500 hover:from-purple-500 hover:to-purple-600 text-white py-3 rounded-xl font-bold shadow-lg shadow-purple-500/20 active:translate-y-0.5 transition-all disabled:opacity-50"
        >
          <div className="flex items-center justify-center gap-2">
            {isLoading ? <RefreshCw className="animate-spin" size={20} /> : <Volume2 size={20} />}
            <span className="text-base">Speak Sentence</span>
          </div>
        </button>

        <div className="flex gap-2">
          <button
            onClick={handleDelete}
            disabled={sentence.length === 0}
            className="flex-1 bg-white hover:bg-zinc-50 text-zinc-700 py-3 rounded-xl font-bold shadow-sm border border-zinc-200/50 flex items-center justify-center gap-2 active:translate-y-0.5 transition-all disabled:opacity-50"
          >
            <Trash2 size={18} />
            <span>Undo Last</span>
          </button>
          <button
            onClick={clearSentence}
            className="flex-1 bg-white hover:bg-zinc-50 text-zinc-700 py-3 rounded-xl font-bold shadow-sm border border-zinc-200/50 flex items-center justify-center gap-2 active:translate-y-0.5 transition-all"
          >
            <RefreshCw size={18} />
            <span>Clear All</span>
          </button>
        </div>
      </div>

      {/* Last Event */}
      {lastEvent && (
        <div className="text-center text-xs font-medium text-white/80 bg-white/10 backdrop-blur-sm rounded-full px-4 py-1.5 mx-auto">
          {lastEvent}
        </div>
      )}

      {/* Sentence Builder */}
      <div className="bg-white/20 backdrop-blur-xl rounded-2xl p-4 shadow-lg min-h-[60px] flex items-center flex-wrap gap-2 border border-white/40 ring-1 ring-white/20 w-full">
        {sentence.length === 0 ? (
          <span className="text-white/60 italic text-base font-medium pl-1">Focus on phrases to build a sentence...</span>
        ) : (
          <AnimatePresence>
            {sentence.map((word, i) => (
              <motion.span
                key={`${word}-${i}`}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                className="text-lg font-bold text-zinc-700 bg-white/90 px-3 py-1 rounded-lg shadow-sm"
              >
                {word}
              </motion.span>
            ))}
          </AnimatePresence>
        )}
        <span className="w-0.5 h-6 bg-white/50 animate-pulse ml-1"></span>
      </div>

      {/* P300 Phrase Grid (6 phrases, 2x3) */}
      <div className="relative p-1 rounded-3xl bg-gradient-to-br from-blue-400 via-purple-400 to-pink-400 shadow-xl mb-4">
        <div className="absolute inset-0 bg-white/40 backdrop-blur-xl rounded-3xl m-[2px]"></div>
        <div className="relative grid grid-cols-2 md:grid-cols-3 gap-3 p-3 min-h-[220px]">
          {phrases.map((phrase, index) => {
            const isActive = flashIndex === index;
            const isSelected = selectedIndex === index;

            return (
              <motion.button
                key={`${phrase}-${index}`}
                onClick={() => handlePhraseClick(index)}
                layout
                className={`
                  relative overflow-hidden rounded-xl text-lg font-bold transition-all duration-75
                  flex items-center justify-center p-3 h-24 md:h-28
                  group shadow-sm border
                  ${isActive
                    ? 'bg-zinc-800 text-white scale-105 z-20 border-zinc-800 shadow-xl'
                    : 'bg-white/80 text-zinc-600 border-white/60 hover:bg-white hover:scale-[1.02]'
                  }
                  ${isSelected
                    ? 'ring-4 ring-emerald-500 bg-emerald-500 text-white z-30 scale-110 shadow-emerald-500/50 border-emerald-500'
                    : ''
                  }
                `}
              >
                <span className="relative z-10 text-center leading-tight">{phrase}</span>

                {isActive && (
                  <motion.div
                    layoutId="flash-overlay"
                    className="absolute inset-0 bg-white/10"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.1 }}
                  />
                )}

                {/* Confidence badge */}
                {isSelected && confidence !== null && (
                  <span className="absolute top-2 right-2 text-[10px] font-mono bg-white/30 px-1.5 py-0.5 rounded-full">
                    {(confidence * 100).toFixed(0)}%
                  </span>
                )}

                <span className={`absolute top-2 left-3 text-[10px] font-mono ${isActive ? 'opacity-50' : 'opacity-30'}`}>
                  {index + 1}
                </span>
              </motion.button>
            );
          })}
        </div>
      </div>

      {/* Demo confirm hint */}
      {isDemoMode && selectedIndex !== null && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex justify-center mb-4"
        >
          <button
            onClick={handleDemoConfirm}
            className="bg-emerald-500 hover:bg-emerald-600 text-white px-6 py-2 rounded-full font-bold shadow-lg shadow-emerald-500/30 transition-all active:scale-95"
          >
            Confirm Selection (Demo Blink)
          </button>
        </motion.div>
      )}
    </div>
  );
};

export default P300Grid;
