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
  startSelection,
  stopSelection,
  doneSend,
  clearSentence as clearSentenceAPI,
  type BCIEvent,
  type SystemStatus,
} from '../services/api';
import {
  Volume2,
  Play,
  StopCircle,
  Send,
  RefreshCw,
  Trash2,
  Wifi,
  WifiOff,
  Eye,
  Zap,
  Check,
  MoreHorizontal,
  Square,
} from 'lucide-react';
import EEGMonitor from './EEGMonitor';
import BandPowerHistogram from './BandPowerHistogram';
import DeepLearningPanel from './DeepLearningPanel';
import LiveTestPanel from './LiveTestPanel';

const FALLBACK_PHRASES = ["yes", "no", "help", "water", "pain", "Other"];
const OTHER_LABEL = "Other";

type SelectionPhase = 'idle' | 'warmup' | 'calibrating' | 'highlighting' | 'confirming' | 'executing' | 'stopped';

const P300Grid: React.FC = () => {
  const [phrases, setPhrases] = useState<string[]>(FALLBACK_PHRASES);
  const [sentence, setSentence] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [lastEvent, setLastEvent] = useState<string>('');

  const [blinkFlash, setBlinkFlash] = useState(false);
  const [clenchFlash, setClenchFlash] = useState(false);

  const [selPhase, setSelPhase] = useState<SelectionPhase>('idle');
  const [warmupProgress, setWarmupProgress] = useState(0);
  const [calBlinks, setCalBlinks] = useState(0);
  const [calNeeded, setCalNeeded] = useState(2);
  const [highlightIndex, setHighlightIndex] = useState<number | null>(null);
  const [confirmedIndex, setConfirmedIndex] = useState<number | null>(null);
  const [confirmedPhrase, setConfirmedPhrase] = useState<string>('');

  const audioRef = useRef<HTMLAudioElement | null>(null);

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

      bciSocket.on('words_updated', (e: BCIEvent) => {
        const { words, phrases: fullPhrases, sentence: newSentence } = e.data;
        if (Array.isArray(fullPhrases) && fullPhrases.length > 0) {
          setPhrases(fullPhrases);
        } else if (Array.isArray(words) && words.length > 0) {
          setPhrases([...words, OTHER_LABEL]);
        }
        if (Array.isArray(newSentence)) {
          setSentence(newSentence);
        }
      }),

      bciSocket.on('word_selected', (e: BCIEvent) => {
        if (Array.isArray(e.data.sentence)) {
          setSentence(e.data.sentence);
        }
        if (e.data.word) {
          setLastEvent(`Added: "${e.data.word}"`);
        }
      }),

      bciSocket.on('sentence_cleared', (e: BCIEvent) => {
        setSentence([]);
        if (e.data.spoken) {
          setLastEvent(`Spoke: "${e.data.spoken}"`);
        }
      }),

      bciSocket.on('session_stopped', () => {
        setSelPhase('stopped');
        setHighlightIndex(null);
        setConfirmedIndex(null);
        setLastEvent('Session stopped');
      }),

      bciSocket.on('blink_detected', () => {
        setBlinkFlash(true);
        setTimeout(() => setBlinkFlash(false), 500);
      }),

      bciSocket.on('clench_detected', () => {
        setClenchFlash(true);
        setTimeout(() => setClenchFlash(false), 500);
      }),

      bciSocket.on('phrase_confirmed', (e: BCIEvent) => {
        if (Array.isArray(e.data.history)) {
          setSentence(e.data.history);
        }
      }),

      bciSocket.on('phrase_deleted', (e: BCIEvent) => {
        if (Array.isArray(e.data.history)) {
          setSentence(e.data.history);
        }
      }),

      bciSocket.on('system_status', (e: BCIEvent) => {
        setStatus(e.data as any);
      }),

      bciSocket.on('warmup_status', (e: BCIEvent) => {
        const { state, progress, message } = e.data;
        if (state === 'warmup') {
          setSelPhase('warmup');
          setWarmupProgress(progress ?? 0);
          setLastEvent(message || 'Stabilizing...');
        } else if (state === 'idle') {
          setSelPhase('idle');
          setHighlightIndex(null);
          setConfirmedIndex(null);
          setLastEvent(message || '');
        }
      }),

      bciSocket.on('calibration_status', (e: BCIEvent) => {
        const { state, blinks_detected, blinks_needed } = e.data;
        if (state === 'calibrating') {
          setSelPhase('calibrating');
          setCalBlinks(blinks_detected ?? 0);
          setCalNeeded(blinks_needed ?? 2);
          setLastEvent(`Calibration: blink ${blinks_detected}/${blinks_needed}`);
        } else if (state === 'complete') {
          setLastEvent('Calibration complete');
        }
      }),

      bciSocket.on('highlight_changed', (e: BCIEvent) => {
        setSelPhase('highlighting');
        setHighlightIndex(e.data.index);
        setConfirmedIndex(null);
      }),

      bciSocket.on('selection_confirmed', (e: BCIEvent) => {
        setSelPhase('confirming');
        setConfirmedIndex(e.data.index);
        setConfirmedPhrase(e.data.phrase || '');
        setHighlightIndex(null);
        setLastEvent(`Selected: "${e.data.phrase}"`);
      }),

      bciSocket.on('selection_executed', (e: BCIEvent) => {
        if (e.data.action === 'done_send') {
          setLastEvent(`Spoke: "${e.data.phrase}"`);
        } else {
          setLastEvent(`Confirmed: "${e.data.phrase}"`);
        }
        setConfirmedIndex(null);
        setHighlightIndex(null);
      }),
    );

    return () => unsubs.forEach((u) => u());
  }, []);

  // ── Actions ────────────────────────────────────────────────

  const handleStartStop = useCallback(async () => {
    if (selPhase === 'idle' || selPhase === 'stopped') {
      try {
        setSelPhase('idle');
        await startSelection();
      } catch (e: any) {
        setLastEvent(`Error: ${e.message}`);
      }
    } else {
      try {
        if (audioRef.current) {
          audioRef.current.pause();
          audioRef.current = null;
        }
        await stopSelection();
        setSelPhase('stopped');
        setHighlightIndex(null);
        setConfirmedIndex(null);
      } catch (e: any) {
        setLastEvent(`Error: ${e.message}`);
      }
    }
  }, [selPhase]);

  const handlePhraseClick = useCallback(async (index: number) => {
    if (selPhase !== 'idle' && selPhase !== 'stopped') return;
    setIsLoading(true);
    try {
      const result = await confirmPhrase(index);
      setSentence(result.history);
      setPhrases(result.new_phrases.length > 0 ? result.new_phrases : FALLBACK_PHRASES);
    } catch {
      const phrase = phrases[index];
      if (phrase !== OTHER_LABEL) {
        setSentence(prev => [...prev, phrase]);
      }
    }
    setIsLoading(false);
  }, [phrases, selPhase]);

  const handleDoneSend = async () => {
    if (sentence.length === 0) return;
    const text = sentence.join(" ");
    setIsSpeaking(true);
    setLastEvent('Speaking...');

    try {
      const audioDataUrl = await speakText(text);
      if (audioDataUrl) {
        const audio = new Audio(audioDataUrl);
        audioRef.current = audio;
        audio.onended = () => {
          setIsSpeaking(false);
          audioRef.current = null;
        };
        audio.play();
      } else {
        setIsSpeaking(false);
      }
    } catch {
      setIsSpeaking(false);
    }

    try {
      await doneSend();
    } catch {
      setSentence([]);
    }
  };

  const handleDelete = async () => {
    try {
      const result = await deleteLastPhrase();
      setSentence(result.history);
    } catch {
      setSentence(prev => prev.slice(0, -1));
    }
  };

  const handleClear = async () => {
    try {
      await clearSentenceAPI();
      setSentence([]);
    } catch {
      setSentence([]);
    }
  };

  const isActive = selPhase !== 'idle' && selPhase !== 'stopped';
  const isStopped = selPhase === 'stopped';

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
          {status?.eegnet_gesture && (
            <span className="text-[10px] font-bold bg-blue-200 text-blue-800 px-2 py-0.5 rounded-full">GESTURE-DL</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <div className={`flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold transition-all duration-200 ${blinkFlash ? 'bg-blue-400 text-white scale-110' : 'bg-blue-100 text-blue-500'}`}>
            <Eye size={10} /> BLINK
          </div>
          <div className={`flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold transition-all duration-200 ${clenchFlash ? 'bg-orange-400 text-white scale-110' : 'bg-orange-100 text-orange-500'}`}>
            <Zap size={10} /> CLENCH
          </div>
        </div>
      </div>

      {/* Start / Stop Button */}
      <div className="flex justify-center py-1">
        <button
          onClick={handleStartStop}
          className={`
            relative group overflow-hidden px-10 py-3.5 rounded-full font-black text-lg text-white shadow-xl transition-all transform hover:scale-105 active:scale-95
            ${isActive
              ? 'bg-gradient-to-b from-red-400 to-red-600 shadow-red-500/40 border-2 border-red-300'
              : isStopped
              ? 'bg-gradient-to-b from-emerald-400 to-emerald-600 shadow-emerald-500/40 border-2 border-emerald-300'
              : 'bg-gradient-to-b from-emerald-400 to-emerald-600 shadow-emerald-500/40 border-2 border-emerald-300'
            }
          `}
        >
          <div className="absolute inset-0 bg-white/20 group-hover:bg-white/30 transition-colors" />
          <div className="absolute top-0 left-0 w-full h-1/2 bg-white/20 rounded-t-full blur-[1px]" />
          <div className="flex items-center gap-3 relative z-10 drop-shadow-md">
            {isActive
              ? <><Square size={24} fill="currentColor" /><span className="tracking-wide">STOP</span></>
              : <><Play size={24} fill="currentColor" /><span className="tracking-wide">START</span></>
            }
          </div>
        </button>
      </div>

      {/* Stopped Overlay */}
      <AnimatePresence>
        {isStopped && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="bg-zinc-100 border-2 border-zinc-400 rounded-2xl p-5 text-center shadow-lg"
          >
            <div className="flex items-center justify-center gap-2 text-zinc-600 font-bold text-lg">
              <StopCircle size={24} className="text-zinc-500" />
              Session Stopped
            </div>
            <p className="text-zinc-500 text-sm mt-1">Press START to begin a new session</p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Warmup Overlay */}
      <AnimatePresence>
        {selPhase === 'warmup' && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="bg-amber-50 border-2 border-amber-300 rounded-2xl p-5 text-center shadow-lg"
          >
            <p className="text-amber-800 font-bold text-lg mb-3">Stabilizing signal...</p>
            <p className="text-amber-600 text-sm mb-3">Stay still and relax</p>
            <div className="w-full bg-amber-200 rounded-full h-3 overflow-hidden">
              <motion.div
                className="h-full bg-amber-500 rounded-full"
                initial={{ width: '0%' }}
                animate={{ width: `${warmupProgress * 100}%` }}
                transition={{ duration: 0.3 }}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Calibration Overlay */}
      <AnimatePresence>
        {selPhase === 'calibrating' && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="bg-blue-50 border-2 border-blue-300 rounded-2xl p-5 text-center shadow-lg"
          >
            <p className="text-blue-800 font-bold text-lg mb-2">Blink to calibrate</p>
            <p className="text-blue-600 text-sm mb-4">Perform {calNeeded} intentional blinks</p>
            <div className="flex justify-center gap-3">
              {Array.from({ length: calNeeded }).map((_, i) => (
                <motion.div
                  key={i}
                  animate={i < calBlinks ? { scale: [1, 1.3, 1] } : {}}
                  transition={{ duration: 0.3 }}
                  className={`w-12 h-12 rounded-full flex items-center justify-center text-xl font-bold border-2 transition-all ${
                    i < calBlinks
                      ? 'bg-blue-500 text-white border-blue-600 shadow-lg shadow-blue-500/40'
                      : 'bg-white text-blue-300 border-blue-200'
                  }`}
                >
                  {i < calBlinks ? <Check size={24} /> : <Eye size={24} />}
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Confirmation Banner */}
      <AnimatePresence>
        {selPhase === 'confirming' && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="bg-emerald-50 border-2 border-emerald-400 rounded-2xl p-4 text-center shadow-lg"
          >
            <div className="flex items-center justify-center gap-2 text-emerald-700 font-bold text-lg">
              <Check size={24} className="text-emerald-500" />
              {confirmedPhrase === OTHER_LABEL ? 'Loading new words...' : `Selected: "${confirmedPhrase}"`}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Word Grid (5 words + Other, 2x3) */}
      <div className="relative p-1 rounded-3xl bg-gradient-to-br from-blue-400 via-purple-400 to-pink-400 shadow-xl">
        <div className="absolute inset-0 bg-white/40 backdrop-blur-xl rounded-3xl m-[2px]"></div>
        <div className="relative grid grid-cols-2 md:grid-cols-3 gap-3 p-3 min-h-[220px]">
          {phrases.map((phrase, index) => {
            const isHighlighted = highlightIndex === index && selPhase === 'highlighting';
            const isConfirmed = confirmedIndex === index && selPhase === 'confirming';
            const isOther = phrase === OTHER_LABEL;

            return (
              <motion.button
                key={`${phrase}-${index}`}
                onClick={() => handlePhraseClick(index)}
                disabled={isActive && selPhase !== 'idle'}
                animate={{
                  scale: isConfirmed ? 1.10 : isHighlighted ? 1.08 : 1,
                }}
                transition={{ type: 'spring', stiffness: 300, damping: 20 }}
                className={`
                  relative overflow-hidden rounded-xl font-bold
                  flex items-center justify-center p-3 h-24 md:h-28
                  group shadow-sm border transition-colors duration-200
                  ${isConfirmed
                    ? 'bg-emerald-500 text-white border-emerald-600 ring-4 ring-emerald-400 shadow-xl shadow-emerald-500/50 z-30'
                    : isHighlighted
                    ? 'bg-amber-400 text-white border-amber-500 ring-4 ring-amber-300 shadow-xl shadow-amber-400/50 z-20'
                    : isStopped
                    ? 'bg-zinc-200 text-zinc-400 border-zinc-300 cursor-not-allowed'
                    : isOther
                    ? 'bg-slate-100 text-slate-600 border-2 border-dashed border-slate-400 hover:bg-slate-200 hover:scale-[1.02]'
                    : isActive
                    ? 'bg-white/60 text-zinc-400 border-white/40'
                    : 'bg-white/80 text-zinc-600 border-white/60 hover:bg-white hover:scale-[1.02]'
                  }
                `}
              >
                <span className={`relative z-10 text-center leading-tight ${isOther ? 'text-base' : 'text-xl'}`}>
                  {isOther ? (
                    <span className="flex items-center gap-2">
                      <MoreHorizontal size={20} />
                      Other
                    </span>
                  ) : (
                    phrase
                  )}
                </span>

                {isConfirmed && (
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    className="absolute top-2 right-2 bg-white/30 rounded-full p-1"
                  >
                    <Check size={16} />
                  </motion.div>
                )}

                <span className={`absolute top-2 left-3 text-[10px] font-mono ${isHighlighted || isConfirmed ? 'opacity-60' : 'opacity-30'}`}>
                  {index + 1}
                </span>

                {isHighlighted && (
                  <motion.div
                    className="absolute bottom-0 left-0 right-0 h-1 bg-white/60"
                    initial={{ scaleX: 0 }}
                    animate={{ scaleX: 1 }}
                    transition={{ duration: 2, ease: 'linear' }}
                    style={{ transformOrigin: 'left' }}
                  />
                )}
              </motion.button>
            );
          })}
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
          <span className="text-white/60 italic text-base font-medium pl-1">Blink to build your sentence...</span>
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

      {/* Control Panel */}
      <div className="bg-white/40 backdrop-blur-md p-4 rounded-3xl border border-white/50 shadow-sm space-y-3">
        <button
          onClick={handleDoneSend}
          disabled={sentence.length === 0 || isLoading || isSpeaking}
          className="w-full group relative overflow-hidden bg-gradient-to-r from-emerald-400 to-teal-500 hover:from-emerald-500 hover:to-teal-600 text-white py-3 rounded-xl font-bold shadow-lg shadow-emerald-500/20 active:translate-y-0.5 transition-all disabled:opacity-50"
        >
          <div className="flex items-center justify-center gap-2">
            {isSpeaking ? <Volume2 className="animate-pulse" size={20} /> : isLoading ? <RefreshCw className="animate-spin" size={20} /> : <Send size={20} />}
            <span className="text-base">{isSpeaking ? 'Speaking...' : 'Done / Send'}</span>
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
            onClick={handleClear}
            disabled={sentence.length === 0}
            className="flex-1 bg-white hover:bg-zinc-50 text-zinc-700 py-3 rounded-xl font-bold shadow-sm border border-zinc-200/50 flex items-center justify-center gap-2 active:translate-y-0.5 transition-all disabled:opacity-50"
          >
            <RefreshCw size={18} />
            <span>Clear All</span>
          </button>
        </div>
      </div>

      {/* EEG Monitor */}
      <div className="bg-white/40 backdrop-blur-md p-2 rounded-2xl border border-white/50 shadow-sm">
        <div className="bg-white/60 rounded-xl p-1">
          <EEGMonitor isFlashing={selPhase === 'highlighting'} />
        </div>
      </div>

      {/* Band Power Histogram */}
      <div className="bg-white/40 backdrop-blur-md p-2 rounded-2xl border border-white/50 shadow-sm">
        <div className="bg-white/60 rounded-xl p-1">
          <BandPowerHistogram />
        </div>
      </div>

      {/* Live Gesture Test */}
      <LiveTestPanel />

      {/* Deep Learning Panel */}
      <DeepLearningPanel />
    </div>
  );
};

export default P300Grid;
