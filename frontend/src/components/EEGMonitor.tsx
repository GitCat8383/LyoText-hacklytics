import React, { useEffect, useState, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, CartesianGrid } from 'recharts';
import { bciSocket, type BCIEvent } from '../services/api';

interface EEGMonitorProps {
  isFlashing: boolean;
}

interface DataPoint {
  time: number;
  af7: number;
  af8: number;
  tp9: number;
  tp10: number;
}

const MAX_POINTS = 60;

const EEGMonitor: React.FC<EEGMonitorProps> = ({ isFlashing }) => {
  const [data, setData] = useState<DataPoint[]>([]);
  const [blinkCount, setBlinkCount] = useState(0);
  const [clenchCount, setClenchCount] = useState(0);
  const [channel, setChannel] = useState<'af7' | 'af8' | 'tp9' | 'tp10'>('af7');
  const timeRef = useRef(0);
  const wsSubscribedRef = useRef(false);

  // Subscribe to EEG stream via WebSocket
  useEffect(() => {
    if (!wsSubscribedRef.current) {
      bciSocket.subscribeEEG();
      wsSubscribedRef.current = true;
    }

    const unsubEEG = bciSocket.on('eeg_sample', (e: BCIEvent) => {
      const sample = e.data;
      timeRef.current += 1;
      setData(prev => {
        const next = [...prev, {
          time: timeRef.current,
          af7: sample.af7 ?? 0,
          af8: sample.af8 ?? 0,
          tp9: sample.tp9 ?? 0,
          tp10: sample.tp10 ?? 0,
        }];
        if (next.length > MAX_POINTS) next.shift();
        return next;
      });
    });

    const unsubBlink = bciSocket.on('blink_detected', () => {
      setBlinkCount(c => c + 1);
    });

    const unsubClench = bciSocket.on('clench_detected', () => {
      setClenchCount(c => c + 1);
    });

    return () => {
      unsubEEG();
      unsubBlink();
      unsubClench();
    };
  }, []);

  // Fallback: simulated data when no backend data arrives
  useEffect(() => {
    if (data.length > 0) return;

    const interval = setInterval(() => {
      timeRef.current += 1;
      const noise = (Math.random() - 0.5) * 10;
      const spike = isFlashing && Math.random() > 0.95 ? 50 : 0;
      setData(prev => {
        const next = [...prev, {
          time: timeRef.current,
          af7: noise + spike,
          af8: noise * 0.9,
          tp9: noise * 1.1,
          tp10: noise * 0.8,
        }];
        if (next.length > MAX_POINTS) next.shift();
        return next;
      });
    }, 100);

    return () => clearInterval(interval);
  }, [isFlashing, data.length]);

  const channels: Array<{ key: 'af7' | 'af8' | 'tp9' | 'tp10'; label: string; color: string }> = [
    { key: 'af7', label: 'AF7', color: '#3b82f6' },
    { key: 'af8', label: 'AF8', color: '#8b5cf6' },
    { key: 'tp9', label: 'TP9', color: '#06b6d4' },
    { key: 'tp10', label: 'TP10', color: '#f59e0b' },
  ];

  const activeChannel = channels.find(c => c.key === channel)!;

  return (
    <div className="w-full h-52 bg-white/90 rounded-2xl p-4 border-4 border-blue-200 shadow-xl backdrop-blur-sm">
      <div className="flex justify-between items-center mb-2">
        <div className="flex items-center gap-3">
          <h3 className="text-blue-500 font-bold text-xs tracking-widest uppercase flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isFlashing ? 'bg-red-500 animate-pulse' : 'bg-blue-400'}`}></div>
            EEG Signal
          </h3>
          {/* Channel selector */}
          <div className="flex gap-1">
            {channels.map(ch => (
              <button
                key={ch.key}
                onClick={() => setChannel(ch.key)}
                className={`text-[9px] font-bold px-1.5 py-0.5 rounded transition-all ${
                  channel === ch.key
                    ? 'bg-blue-500 text-white'
                    : 'bg-zinc-100 text-zinc-400 hover:bg-zinc-200'
                }`}
              >
                {ch.label}
              </button>
            ))}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[9px] font-bold text-blue-400">
            👁 {blinkCount} | 💪 {clenchCount}
          </span>
          <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${isFlashing ? 'bg-red-100 text-red-500' : 'bg-blue-100 text-blue-500'}`}>
            {isFlashing ? 'ACTIVE' : 'IDLE'}
          </span>
        </div>
      </div>
      <ResponsiveContainer width="100%" height="85%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#93c5fd" opacity={0.3} />
          <YAxis domain={[-80, 80]} hide />
          <Line
            type="monotone"
            dataKey={channel}
            stroke={isFlashing ? "#ef4444" : activeChannel.color}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default EEGMonitor;
