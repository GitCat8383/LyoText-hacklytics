/**
 * REST + WebSocket client for the Hacklytic FastAPI backend.
 */

const API_BASE = "/api";
const WS_URL =
  typeof window !== "undefined"
    ? `ws://${window.location.hostname}:8000/ws/events`
    : "ws://localhost:8000/ws/events";

// ── REST helpers ─────────────────────────────────────────────

async function fetchJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

// ── Types ────────────────────────────────────────────────────

export interface SystemStatus {
  eeg_connected: boolean;
  classifier_loaded: boolean;
  classifier_calibrating: boolean;
  calibration_epochs: number;
  redis_connected: boolean;
  simulate_mode: boolean;
}

export interface BCIEvent {
  type: string;
  data: Record<string, any>;
  ts: number;
}

export interface P300Result {
  selected_index: number;
  confidence: number;
  phrase: string;
}

export interface EEGSample {
  tp9: number;
  af7: number;
  af8: number;
  tp10: number;
}

// ── REST API ─────────────────────────────────────────────────

export async function getStatus(): Promise<SystemStatus> {
  return fetchJSON("/status");
}

export async function getPhrases(): Promise<string[]> {
  const data = await fetchJSON<{ phrases: string[] }>("/phrases");
  return data.phrases;
}

export async function confirmPhrase(index: number): Promise<{
  confirmed: string;
  history: string[];
  new_phrases: string[];
}> {
  return fetchJSON(`/phrases/confirm/${index}`, { method: "POST" });
}

export async function getHistory(): Promise<string[]> {
  const data = await fetchJSON<{ history: string[] }>("/history");
  return data.history;
}

export async function deleteLastPhrase(): Promise<{
  removed: string;
  history: string[];
}> {
  return fetchJSON("/history/last", { method: "DELETE" });
}

export async function startCalibration(): Promise<void> {
  await fetchJSON("/calibration/start", { method: "POST" });
}

export async function stopCalibration(): Promise<{ accuracy: number }> {
  return fetchJSON("/calibration/stop", { method: "POST" });
}

export async function getConfig(): Promise<Record<string, any>> {
  return fetchJSON("/config");
}

export async function updateConfig(
  updates: Record<string, any>
): Promise<void> {
  await fetchJSON("/config", {
    method: "PATCH",
    body: JSON.stringify(updates),
  });
}

// ── WebSocket ────────────────────────────────────────────────

type EventCallback = (event: BCIEvent) => void;

export class BCIWebSocket {
  private ws: WebSocket | null = null;
  private listeners = new Map<string, Set<EventCallback>>();
  private globalListeners = new Set<EventCallback>();
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private _connected = false;

  get connected(): boolean {
    return this._connected;
  }

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    try {
      this.ws = new WebSocket(WS_URL);

      this.ws.onopen = () => {
        this._connected = true;
        this.emit({ type: "ws_connected", data: {}, ts: Date.now() / 1000 });
      };

      this.ws.onclose = () => {
        this._connected = false;
        this.emit({
          type: "ws_disconnected",
          data: {},
          ts: Date.now() / 1000,
        });
        this.scheduleReconnect();
      };

      this.ws.onerror = () => {
        this._connected = false;
      };

      this.ws.onmessage = (msg) => {
        try {
          const event: BCIEvent = JSON.parse(msg.data);
          this.emit(event);
        } catch {
          // ignore malformed messages
        }
      };
    } catch {
      this.scheduleReconnect();
    }
  }

  disconnect(): void {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.ws?.close();
    this.ws = null;
    this._connected = false;
  }

  /** Subscribe to a specific event type. */
  on(eventType: string, callback: EventCallback): () => void {
    if (!this.listeners.has(eventType)) {
      this.listeners.set(eventType, new Set());
    }
    this.listeners.get(eventType)!.add(callback);
    return () => this.listeners.get(eventType)?.delete(callback);
  }

  /** Subscribe to all events. */
  onAny(callback: EventCallback): () => void {
    this.globalListeners.add(callback);
    return () => this.globalListeners.delete(callback);
  }

  /** Request real-time EEG sample streaming (high bandwidth). */
  subscribeEEG(): void {
    this.send({ command: "subscribe_eeg" });
  }

  unsubscribeEEG(): void {
    this.send({ command: "unsubscribe_eeg" });
  }

  private send(data: Record<string, any>): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  private emit(event: BCIEvent): void {
    this.globalListeners.forEach((cb) => {
      try {
        cb(event);
      } catch {
        /* ignore */
      }
    });
    this.listeners.get(event.type)?.forEach((cb) => {
      try {
        cb(event);
      } catch {
        /* ignore */
      }
    });
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) return;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, 3000);
  }
}

export const bciSocket = new BCIWebSocket();
