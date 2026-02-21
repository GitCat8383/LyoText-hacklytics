import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
  bciSocket,
  startLiveTest,
  stopLiveTest,
  getLiveTestStatus,
  type BCIEvent,
} from '../services/api';
import { Eye, Zap, Circle, Activity, Play, StopCircle } from 'lucide-react';

interface Prediction {
  class_name: string;
  confidence: number;
  timestamp: number;
}

const GESTURE_CONFIG: Record<string, { icon: React.ReactNode; color: string; bg: string; label: string }> = {
  idle: {
    icon: <Circle size={48} />,
    color: 'text-gray-400',
    bg: 'from-gray-100 to-gray-200',
    label: 'Idle',
  },
  blink: {
    icon: <Eye size={48} />,
    color: 'text-blue-500',
    bg: 'from-blue-100 to-blue-300',
    label: 'Blink',
  },
  clench: {
    icon: <Zap size={48} />,
    color: 'text-orange-500',
    bg: 'from-orange-100 to-orange-300',
    label: 'Jaw Clench',
  },
};

const MAX_HISTORY = 30;

const LiveTestPanel: React.FC = () => {
  const [active, setActive] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [current, setCurrent] = useState<Prediction | null>(null);
  const [history, setHistory] = useState<Prediction[]>([]);
  const [flash, setFlash] = useState(false);
  const flashTimeout = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => {
    getLiveTestStatus()
      .then((s) => {
        setActive(s.active);
        setModelLoaded(s.model_loaded);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    const unsub = bciSocket.on('gesture_prediction', (e: BCIEvent) => {
      const pred: Prediction = {
        class_name: e.data.class_name,
        confidence: e.data.confidence,
        timestamp: e.ts,
      };
      setCurrent(pred);

      if (pred.class_name !== 'idle' && pred.confidence >= 0.7) {
        setHistory((prev) => [pred, ...prev].slice(0, MAX_HISTORY));
        setFlash(true);
        if (flashTimeout.current) clearTimeout(flashTimeout.current);
        flashTimeout.current = setTimeout(() => setFlash(false), 600);
      }
    });

    return unsub;
  }, []);

  const handleToggle = useCallback(async () => {
    try {
      if (active) {
        await stopLiveTest();
        setActive(false);
        setCurrent(null);
      } else {
        await startLiveTest();
        setActive(true);
        setHistory([]);
      }
    } catch (err: any) {
      console.error('Live test toggle failed:', err);
    }
  }, [active]);

  const gestureInfo = GESTURE_CONFIG[current?.class_name ?? 'idle'] ?? GESTURE_CONFIG.idle;
  const isGesture = current && current.class_name !== 'idle' && current.confidence >= 0.7;

  return (
    <div className="bg-white/80 backdrop-blur-md rounded-2xl border-2 border-indigo-300 shadow-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-gradient-to-r from-indigo-50 to-purple-50 border-b border-indigo-200">
        <div className="flex items-center gap-2">
          <Activity size={18} className="text-indigo-600" />
          <span className="font-extrabold text-indigo-800 text-sm">Live EEG Gesture Test</span>
          {active && (
            <span className="flex items-center gap-1 text-[10px] font-bold text-green-600 bg-green-100 px-2 py-0.5 rounded-full">
              <span className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse" />
              LIVE
            </span>
          )}
        </div>
        <button
          onClick={handleToggle}
          disabled={!modelLoaded}
          className={`
            flex items-center gap-1.5 px-4 py-1.5 rounded-full font-bold text-xs text-white
            transition-all transform hover:scale-105 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed
            ${active
              ? 'bg-gradient-to-r from-red-400 to-red-500 shadow-red-300/50 shadow-md'
              : 'bg-gradient-to-r from-green-400 to-emerald-500 shadow-green-300/50 shadow-md'
            }
          `}
        >
          {active ? <StopCircle size={14} /> : <Play size={14} />}
          {active ? 'Stop Test' : 'Start Test'}
        </button>
      </div>

      {!modelLoaded && (
        <div className="px-4 py-3 text-sm text-amber-700 bg-amber-50 border-b border-amber-200">
          No gesture model loaded. Train or load a model first.
        </div>
      )}

      {active && (
        <div className="p-4 space-y-4">
          {/* Main prediction display */}
          <div className="relative">
            <motion.div
              animate={flash ? { scale: [1, 1.03, 1] } : {}}
              transition={{ duration: 0.3 }}
              className={`
                relative flex flex-col items-center justify-center py-8 rounded-2xl
                bg-gradient-to-b ${gestureInfo.bg}
                border-2 ${isGesture ? 'border-indigo-400 shadow-xl' : 'border-gray-200 shadow-sm'}
                transition-all duration-300
              `}
            >
              {/* Pulse ring on gesture detection */}
              <AnimatePresence>
                {isGesture && (
                  <motion.div
                    key="pulse"
                    initial={{ scale: 0.8, opacity: 0.8 }}
                    animate={{ scale: 2.5, opacity: 0 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.8 }}
                    className={`absolute w-20 h-20 rounded-full ${
                      current?.class_name === 'blink' ? 'bg-blue-400' : 'bg-orange-400'
                    }`}
                  />
                )}
              </AnimatePresence>

              <motion.div
                key={current?.class_name ?? 'idle'}
                initial={{ scale: 0.5, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ type: 'spring', stiffness: 300, damping: 20 }}
                className={`relative z-10 ${gestureInfo.color}`}
              >
                {gestureInfo.icon}
              </motion.div>

              <motion.span
                key={`label-${current?.class_name}`}
                initial={{ y: 10, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                className="relative z-10 mt-3 text-2xl font-black tracking-tight text-gray-800"
              >
                {gestureInfo.label}
              </motion.span>

              {/* Confidence bar */}
              {current && (
                <div className="relative z-10 mt-3 w-48">
                  <div className="flex justify-between text-[10px] font-bold text-gray-500 mb-1">
                    <span>Confidence</span>
                    <span>{(current.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-2.5 bg-white/60 rounded-full overflow-hidden shadow-inner">
                    <motion.div
                      animate={{ width: `${current.confidence * 100}%` }}
                      transition={{ duration: 0.2 }}
                      className={`h-full rounded-full ${
                        current.confidence >= 0.9
                          ? 'bg-green-400'
                          : current.confidence >= 0.7
                          ? 'bg-blue-400'
                          : 'bg-gray-300'
                      }`}
                    />
                  </div>
                </div>
              )}
            </motion.div>
          </div>

          {/* Detection history */}
          {history.length > 0 && (
            <div>
              <span className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">
                Recent Detections
              </span>
              <div className="mt-1.5 flex flex-wrap gap-1.5">
                <AnimatePresence mode="popLayout">
                  {history.slice(0, 15).map((p, i) => {
                    const cfg = GESTURE_CONFIG[p.class_name] ?? GESTURE_CONFIG.idle;
                    return (
                      <motion.div
                        key={`${p.timestamp}-${i}`}
                        initial={{ scale: 0, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        exit={{ scale: 0, opacity: 0 }}
                        layout
                        className={`
                          flex items-center gap-1 px-2 py-1 rounded-full text-[11px] font-bold
                          ${p.class_name === 'blink'
                            ? 'bg-blue-100 text-blue-700'
                            : 'bg-orange-100 text-orange-700'
                          }
                        `}
                      >
                        {React.cloneElement(cfg.icon as React.ReactElement, { size: 12 })}
                        {cfg.label}
                        <span className="opacity-60">{(p.confidence * 100).toFixed(0)}%</span>
                      </motion.div>
                    );
                  })}
                </AnimatePresence>
              </div>
            </div>
          )}
        </div>
      )}

      {!active && modelLoaded && (
        <div className="px-4 py-6 text-center text-sm text-gray-500">
          Press <span className="font-bold text-indigo-600">Start Test</span> to see real-time EEG gesture classification.
          <br />
          <span className="text-xs text-gray-400 mt-1 block">Blink, clench your jaw, or stay idle — the model classifies every 500ms.</span>
        </div>
      )}
    </div>
  );
};

export default LiveTestPanel;
