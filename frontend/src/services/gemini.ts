/**
 * TTS service — uses Gemini TTS when API key is available,
 * falls back to browser SpeechSynthesis otherwise.
 */

import { GoogleGenAI, Modality } from "@google/genai";

const apiKey = process.env.GEMINI_API_KEY;
const hasGeminiKey = apiKey && apiKey !== "your-gemini-api-key-here";
const ai = hasGeminiKey ? new GoogleGenAI({ apiKey }) : null;

async function geminiSpeak(text: string): Promise<string | null> {
  if (!ai) return null;
  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash-preview-tts",
      contents: [{ parts: [{ text }] }],
      config: {
        responseModalities: [Modality.AUDIO],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: { voiceName: "Kore" },
          },
        },
      },
    });

    const base64Audio =
      response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
    if (base64Audio) {
      return `data:audio/mp3;base64,${base64Audio}`;
    }
    return null;
  } catch (error) {
    console.error("Gemini TTS error:", error);
    return null;
  }
}

function browserSpeak(text: string): Promise<void> {
  return new Promise((resolve) => {
    if (!window.speechSynthesis) {
      resolve();
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.95;
    utterance.pitch = 1;
    utterance.onend = () => resolve();
    utterance.onerror = () => resolve();
    window.speechSynthesis.speak(utterance);
  });
}

/**
 * Speak text aloud. Returns an audio data URL if Gemini TTS is used,
 * or "browser" if the browser fallback was used, or null on failure.
 */
export async function speakText(text: string): Promise<string | null> {
  if (hasGeminiKey) {
    const result = await geminiSpeak(text);
    if (result) return result;
  }

  // Fallback to browser built-in speech
  await browserSpeak(text);
  return "browser";
}
