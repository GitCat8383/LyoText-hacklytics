import React, { useState, useEffect, useCallback } from 'react';
import {
  bciSocket,
  getDataSessions,
  getTrainStatus,
  startCollection,
  stopCollection,
  addManualEpoch,
  saveManualEpochs,
  startTraining,
  reloadModels,
  type BCIEvent,
  type DataSession,
  type TrainStatus,
} from '../services/api';
import {
  BrainCircuit,
  Play,
  Square,
  Database,
  Dumbbell,
  Hand,
  Eye,
  Zap,
  Minus,
  Save,
  RefreshCw,
  Check,
  Loader2,
} from 'lucide-react';

const GESTURE_ICONS: Record<string, React.ReactNode> = {
  idle: <Minus size={14} />,
  blink: <Eye size={14} />,
  clench: <Zap size={14} />,
};

const GESTURE_COLORS: Record<string, string> = {
  idle: 'bg-gray-100 text-gray-600 hover:bg-gray-200',
  blink: 'bg-blue-100 text-blue-600 hover:bg-blue-200',
  clench: 'bg-orange-100 text-orange-600 hover:bg-orange-200',
};

const DeepLearningPanel: React.FC = () => {
  const [expanded, setExpanded] = useState(false);
  const [sessions, setSessions] = useState<DataSession[]>([]);
  const [trainStatus, setTrainStatus] = useState<TrainStatus | null>(null);
  const [collecting, setCollecting] = useState(false);
  const [training, setTraining] = useState(false);
  const [manualCount, setManualCount] = useState(0);
  const [lastMessage, setLastMessage] = useState('');
  const [sessionName, setSessionName] = useState('');

  // Training progress
  const [trainProgress, setTrainProgress] = useState<{
    epoch: number;
    maxEpochs: number;
    trainAcc: number;
    valAcc: number;
    model: string;
  } | null>(null);

  // Collection progress
  const [collectProgress, setCollectProgress] = useState<{
    gesture: string;
    trial: number;
    total: number;
    phase: string;
    message: string;
  } | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [sess, ts] = await Promise.all([
        getDataSessions(),
        getTrainStatus(),
      ]);
      setSessions(sess);
      setTrainStatus(ts);
      setTraining(ts.training);
    } catch {
      /* backend might not be ready */
    }
  }, []);

  useEffect(() => {
    if (expanded) refresh();
  }, [expanded, refresh]);

  // Listen for calibration/training progress events
  useEffect(() => {
    const unsubs = [
      bciSocket.on('calibration_progress', (e: BCIEvent) => {
        const d = e.data;
        if (d.model && d.epoch) {
          setTrainProgress({
            epoch: d.epoch,
            maxEpochs: d.max_epochs,
            trainAcc: d.train_acc,
            valAcc: d.val_acc,
            model: d.model,
          });
        }
        if (d.phase) {
          setCollectProgress({
            gesture: d.gesture || '',
            trial: d.trial || 0,
            total: d.total || 0,
            phase: d.phase,
            message: d.message || '',
          });
        }
        if (d.countdown) {
          setCollectProgress({
            gesture: '',
            trial: 0,
            total: 0,
            phase: 'countdown',
            message: d.message || '',
          });
        }
      }),
      bciSocket.on('system_status', (e: BCIEvent) => {
        if (e.data.status === 'collection_complete') {
          setCollecting(false);
          setCollectProgress(null);
          setLastMessage(`Collection done: ${e.data.total_epochs} epochs saved`);
          refresh();
        }
        if (e.data.status === 'collection_started') {
          setCollecting(true);
        }
      }),
    ];
    return () => unsubs.forEach((u) => u());
  }, [refresh]);

  const handleStartCollection = async () => {
    const name = sessionName.trim() || `session_${Date.now()}`;
    try {
      await startCollection(name, ['idle', 'blink', 'clench'], 30);
      setCollecting(true);
      setLastMessage('Collection started — follow cues');
    } catch (e: any) {
      setLastMessage(`Error: ${e.message}`);
    }
  };

  const handleStopCollection = async () => {
    try {
      const result = await stopCollection();
      setCollecting(false);
      setCollectProgress(null);
      setLastMessage(`Saved ${result.total_epochs} epochs`);
      refresh();
    } catch (e: any) {
      setLastMessage(`Error: ${e.message}`);
    }
  };

  const handleManualLabel = async (label: string) => {
    try {
      const result = await addManualEpoch(label);
      setManualCount(result.total_collected);
      setLastMessage(`${label} epoch captured (${result.total_collected} total)`);
    } catch (e: any) {
      setLastMessage(`Error: ${e.message}`);
    }
  };

  const handleSaveManual = async () => {
    try {
      const name = sessionName.trim() || `manual_${Date.now()}`;
      const result = await saveManualEpochs(name);
      setManualCount(0);
      setLastMessage(`Saved ${result.total_epochs} manual epochs`);
      refresh();
    } catch (e: any) {
      setLastMessage(`Error: ${e.message}`);
    }
  };

  const handleTrain = async (modelType: 'p300' | 'gesture') => {
    try {
      await startTraining(modelType);
      setTraining(true);
      setTrainProgress(null);
      setLastMessage(`Training ${modelType} model...`);
    } catch (e: any) {
      setLastMessage(`Error: ${e.message}`);
    }
  };

  const handleReload = async () => {
    try {
      await reloadModels();
      await refresh();
      setLastMessage('Models reloaded');
    } catch (e: any) {
      setLastMessage(`Error: ${e.message}`);
    }
  };

  const totalEpochs = sessions.reduce((sum, s) => sum + s.n_epochs, 0);

  return (
    <div className="bg-white/80 backdrop-blur-md rounded-2xl border-2 border-purple-300 shadow-md overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-3.5 bg-gradient-to-r from-purple-50 to-indigo-50 hover:from-purple-100 hover:to-indigo-100 transition-colors cursor-pointer"
      >
        <div className="flex items-center gap-2">
          <BrainCircuit size={20} className="text-purple-600" />
          <span className="text-sm font-extrabold text-purple-800">EEGNet Deep Learning</span>
        </div>
        <div className="flex items-center gap-2">
          {trainStatus?.p300_loaded && (
            <span className="text-[10px] font-bold bg-green-200 text-green-800 px-2 py-0.5 rounded-full">
              P300
            </span>
          )}
          {trainStatus?.gesture_loaded && (
            <span className="text-[10px] font-bold bg-blue-200 text-blue-800 px-2 py-0.5 rounded-full">
              GESTURE
            </span>
          )}
          {training && (
            <Loader2 size={14} className="animate-spin text-purple-500" />
          )}
          <span className="text-zinc-400 text-xs">{expanded ? '▲' : '▼'}</span>
        </div>
      </button>

      {expanded && (
        <div className="px-4 pb-4 space-y-4">
          {/* Status message */}
          {lastMessage && (
            <div className="text-xs text-center text-zinc-600 bg-zinc-100 rounded-lg px-3 py-1.5">
              {lastMessage}
            </div>
          )}

          {/* ── Data Collection ── */}
          <div className="space-y-2">
            <h3 className="text-xs font-bold text-zinc-500 uppercase tracking-wider flex items-center gap-1.5">
              <Database size={12} /> Data Collection
            </h3>

            <input
              type="text"
              value={sessionName}
              onChange={(e) => setSessionName(e.target.value)}
              placeholder="Session name (optional)"
              className="w-full text-sm px-3 py-1.5 rounded-lg bg-white/70 border border-zinc-200 focus:outline-none focus:ring-2 focus:ring-purple-300"
            />

            {/* Guided collection */}
            <div className="flex gap-2">
              {!collecting ? (
                <button
                  onClick={handleStartCollection}
                  className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg bg-purple-500 text-white text-sm font-bold hover:bg-purple-600 transition-colors"
                >
                  <Play size={14} /> Start Guided Collection
                </button>
              ) : (
                <button
                  onClick={handleStopCollection}
                  className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg bg-red-500 text-white text-sm font-bold hover:bg-red-600 transition-colors"
                >
                  <Square size={14} /> Stop Collection
                </button>
              )}
            </div>

            {/* Collection progress */}
            {collecting && collectProgress && (
              <div className="bg-purple-50 rounded-lg p-3 space-y-1.5">
                <div className="text-sm font-bold text-purple-700 text-center">
                  {collectProgress.message}
                </div>
                {collectProgress.total > 0 && (
                  <>
                    <div className="flex justify-between text-xs text-purple-500">
                      <span>Trial {collectProgress.trial}/{collectProgress.total}</span>
                      <span className="uppercase font-bold">{collectProgress.gesture}</span>
                    </div>
                    <div className="w-full bg-purple-200 rounded-full h-1.5">
                      <div
                        className="bg-purple-500 h-1.5 rounded-full transition-all duration-300"
                        style={{
                          width: `${(collectProgress.trial / collectProgress.total) * 100}%`,
                        }}
                      />
                    </div>
                  </>
                )}
              </div>
            )}

            {/* Manual labeling */}
            <div className="space-y-1.5">
              <span className="text-xs text-zinc-500 font-medium">Quick Label (manual):</span>
              <div className="flex gap-1.5">
                {(['idle', 'blink', 'clench'] as const).map((gesture) => (
                  <button
                    key={gesture}
                    onClick={() => handleManualLabel(gesture)}
                    className={`flex-1 flex items-center justify-center gap-1 px-2 py-1.5 rounded-lg text-xs font-bold transition-colors ${GESTURE_COLORS[gesture]}`}
                  >
                    {GESTURE_ICONS[gesture]}
                    {gesture.toUpperCase()}
                  </button>
                ))}
              </div>
              {manualCount > 0 && (
                <div className="flex items-center justify-between">
                  <span className="text-xs text-zinc-500">
                    {manualCount} epochs collected
                  </span>
                  <button
                    onClick={handleSaveManual}
                    className="flex items-center gap-1 px-2 py-1 rounded-md bg-green-100 text-green-700 text-xs font-bold hover:bg-green-200"
                  >
                    <Save size={12} /> Save
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* ── Saved Sessions ── */}
          {sessions.length > 0 && (
            <div className="space-y-2">
              <h3 className="text-xs font-bold text-zinc-500 uppercase tracking-wider flex items-center gap-1.5">
                <Database size={12} /> Saved Data ({totalEpochs} epochs)
              </h3>
              <div className="space-y-1 max-h-32 overflow-y-auto">
                {sessions.map((s) => (
                  <div
                    key={s.name}
                    className="flex items-center justify-between bg-white/70 rounded-lg px-3 py-1.5 text-xs"
                  >
                    <span className="font-medium text-zinc-700 truncate max-w-[140px]">
                      {s.name}
                    </span>
                    <div className="flex items-center gap-2">
                      <span className="text-zinc-400">{s.n_epochs} epochs</span>
                      {s.class_distribution && (
                        <div className="flex gap-1">
                          {Object.entries(s.class_distribution).map(([cls, n]) => (
                            <span
                              key={cls}
                              className="bg-zinc-100 text-zinc-500 px-1 py-0.5 rounded text-[10px]"
                            >
                              {cls}:{n}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ── Training ── */}
          <div className="space-y-2">
            <h3 className="text-xs font-bold text-zinc-500 uppercase tracking-wider flex items-center gap-1.5">
              <Dumbbell size={12} /> Train EEGNet
            </h3>
            <div className="flex gap-2">
              <button
                onClick={() => handleTrain('gesture')}
                disabled={training || totalEpochs < 20}
                className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg bg-blue-500 text-white text-sm font-bold hover:bg-blue-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Hand size={14} />
                Train Gesture
              </button>
              <button
                onClick={() => handleTrain('p300')}
                disabled={training || totalEpochs < 20}
                className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg bg-green-500 text-white text-sm font-bold hover:bg-green-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <BrainCircuit size={14} />
                Train P300
              </button>
            </div>
            {totalEpochs < 20 && totalEpochs > 0 && (
              <p className="text-[10px] text-amber-600 text-center">
                Need at least 20 epochs to train (have {totalEpochs})
              </p>
            )}

            {/* Training progress */}
            {training && trainProgress && (
              <div className="bg-blue-50 rounded-lg p-3 space-y-1.5">
                <div className="flex justify-between text-xs text-blue-600 font-medium">
                  <span>Training {trainProgress.model}...</span>
                  <span>
                    Epoch {trainProgress.epoch}/{trainProgress.maxEpochs}
                  </span>
                </div>
                <div className="w-full bg-blue-200 rounded-full h-1.5">
                  <div
                    className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
                    style={{
                      width: `${(trainProgress.epoch / trainProgress.maxEpochs) * 100}%`,
                    }}
                  />
                </div>
                <div className="flex justify-between text-[10px] text-blue-500">
                  <span>Train acc: {(trainProgress.trainAcc * 100).toFixed(1)}%</span>
                  <span>Val acc: {(trainProgress.valAcc * 100).toFixed(1)}%</span>
                </div>
              </div>
            )}
          </div>

          {/* ── Models ── */}
          <div className="flex items-center justify-between pt-1">
            <div className="flex gap-2">
              {trainStatus?.p300_loaded && (
                <div className="flex items-center gap-1 text-[10px] font-bold text-green-700 bg-green-100 px-2 py-1 rounded-full">
                  <Check size={10} /> P300 Model
                </div>
              )}
              {trainStatus?.gesture_loaded && (
                <div className="flex items-center gap-1 text-[10px] font-bold text-blue-700 bg-blue-100 px-2 py-1 rounded-full">
                  <Check size={10} /> Gesture Model
                </div>
              )}
            </div>
            <button
              onClick={handleReload}
              className="flex items-center gap-1 px-2 py-1 rounded-md bg-zinc-100 text-zinc-600 text-xs hover:bg-zinc-200"
            >
              <RefreshCw size={12} /> Reload
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default DeepLearningPanel;
