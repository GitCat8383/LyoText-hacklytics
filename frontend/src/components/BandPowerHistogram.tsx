import React, { useEffect, useState, useRef } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { bciSocket, type BCIEvent } from '../services/api';

interface BandPowers {
  delta: number;
  theta: number;
  alpha: number;
  beta: number;
  gamma: number;
}

interface PerChannel {
  [ch: string]: BandPowers;
}

const BAND_COLORS: Record<string, string> = {
  delta: '#6366f1',
  theta: '#3b82f6',
  alpha: '#06b6d4',
  beta:  '#10b981',
  gamma: '#f59e0b',
};

const BAND_RANGES: Record<string, string> = {
  delta: '0.5–4 Hz',
  theta: '4–8 Hz',
  alpha: '8–13 Hz',
  beta:  '13–30 Hz',
  gamma: '30–45 Hz',
};

const CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10'];

const CustomTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null;
  const { band, power, range } = payload[0].payload;
  return (
    <div className="bg-white/90 backdrop-blur-sm border border-zinc-200 rounded-lg px-3 py-2 text-xs shadow-lg">
      <p className="font-bold text-zinc-700 capitalize">{band}</p>
      <p className="text-zinc-500">{range}</p>
      <p className="font-mono text-zinc-800">{power.toFixed(3)} µV²/Hz</p>
    </div>
  );
};

const BandPowerHistogram: React.FC = () => {
  const [bands, setBands] = useState<BandPowers | null>(null);
  const [perChannel, setPerChannel] = useState<PerChannel | null>(null);
  const [selectedChannel, setSelectedChannel] = useState<string>('avg');
  const [lastUpdate, setLastUpdate] = useState<number>(0);
  const animFrameRef = useRef<number | null>(null);
  const pendingRef = useRef<{ bands: BandPowers; perChannel: PerChannel } | null>(null);

  useEffect(() => {
    const unsub = bciSocket.on('band_power', (e: BCIEvent) => {
      pendingRef.current = {
        bands: e.data.bands as BandPowers,
        perChannel: e.data.per_channel as PerChannel,
      };
      if (animFrameRef.current === null) {
        animFrameRef.current = requestAnimationFrame(() => {
          animFrameRef.current = null;
          if (pendingRef.current) {
            setBands(pendingRef.current.bands);
            setPerChannel(pendingRef.current.perChannel);
            setLastUpdate(Date.now());
            pendingRef.current = null;
          }
        });
      }
    });
    return () => {
      unsub();
      if (animFrameRef.current !== null) cancelAnimationFrame(animFrameRef.current);
    };
  }, []);

  const activeBands: BandPowers =
    selectedChannel === 'avg' || !perChannel
      ? bands ?? { delta: 0, theta: 0, alpha: 0, beta: 0, gamma: 0 }
      : (perChannel[selectedChannel] ?? { delta: 0, theta: 0, alpha: 0, beta: 0, gamma: 0 });

  const chartData = Object.entries(activeBands).map(([band, power]) => ({
    band,
    power,
    range: BAND_RANGES[band],
  }));

  const isLive = Date.now() - lastUpdate < 2000;

  return (
    <div className="w-full bg-white/90 rounded-2xl p-4 border-4 border-purple-200 shadow-xl backdrop-blur-sm">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isLive ? 'bg-emerald-400 animate-pulse' : 'bg-zinc-300'}`} />
          <h3 className="text-purple-500 font-bold text-xs tracking-widest uppercase">
            Band Power
          </h3>
          <span className="text-[9px] text-zinc-400 font-mono">µV²/Hz</span>
        </div>

        {/* Channel selector */}
        <div className="flex gap-1">
          <button
            onClick={() => setSelectedChannel('avg')}
            className={`text-[9px] font-bold px-1.5 py-0.5 rounded transition-all ${
              selectedChannel === 'avg'
                ? 'bg-purple-500 text-white'
                : 'bg-zinc-100 text-zinc-400 hover:bg-zinc-200'
            }`}
          >
            AVG
          </button>
          {CHANNELS.map(ch => (
            <button
              key={ch}
              onClick={() => setSelectedChannel(ch)}
              className={`text-[9px] font-bold px-1.5 py-0.5 rounded transition-all ${
                selectedChannel === ch
                  ? 'bg-purple-500 text-white'
                  : 'bg-zinc-100 text-zinc-400 hover:bg-zinc-200'
              }`}
            >
              {ch}
            </button>
          ))}
        </div>
      </div>

      {/* Bar chart */}
      <ResponsiveContainer width="100%" height={140}>
        <BarChart data={chartData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e9d5ff" opacity={0.5} vertical={false} />
          <XAxis
            dataKey="band"
            tick={{ fontSize: 10, fontWeight: 700, textTransform: 'capitalize', fill: '#6b7280' }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            tick={{ fontSize: 9, fill: '#9ca3af' }}
            axisLine={false}
            tickLine={false}
            width={40}
            tickFormatter={(v: number) => v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(1)}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(139,92,246,0.08)' }} />
          <Bar dataKey="power" radius={[6, 6, 0, 0]} maxBarSize={52}>
            {chartData.map(entry => (
              <Cell key={entry.band} fill={BAND_COLORS[entry.band]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Band legend */}
      <div className="flex justify-around mt-2">
        {Object.entries(BAND_COLORS).map(([band, color]) => (
          <div key={band} className="flex flex-col items-center gap-0.5">
            <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: color }} />
            <span className="text-[8px] font-bold text-zinc-500 capitalize">{band}</span>
            <span className="text-[7px] text-zinc-400">{BAND_RANGES[band]}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default BandPowerHistogram;
