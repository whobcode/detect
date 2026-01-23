import { useRef, useState } from 'react';

interface AnalysisResult {
  emotionalState: Record<string, number>;
  dominantEmotion: string;
  deceptionScore: number;
  riskLevel: 'low' | 'medium' | 'high';
  explanation: string;
  transcription?: string;
  audioFeatures?: {
    stressIndicators: {
      jitter: number;
      shimmer: number;
      nhr: number;
      stressScore: number;
    };
    pitchAnalysis: {
      meanF0: number;
      f0Range: number;
      vibrato: number;
      vibratoDuration: number;
    };
    pauseDetection: {
      pauseCount: number;
      meanPauseDuration: number;
      pauseFrequency: number;
      speechRate: number;
    };
    formants: {
      f1: number;
      f2: number;
      f3: number;
    };
    spectralFeatures: {
      mfcc: number[];
      spectralCentroid: number;
      spectralRolloff: number;
      zcr: number;
    };
  };
  deceptionFactors?: {
    stressLevel: number;
    hesitationPattern: number;
    pitchVariation: number;
    speechRate: number;
    pauseFrequency: number;
    formantVariability: number;
  };
}

export function Analyzer() {
  const [recording, setRecording] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<AnalysisResult | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const startRecording = async () => {
    setError(null);
    setResults(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        chunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(chunksRef.current, { type: 'audio/webm' });
        stream.getTracks().forEach((track) => track.stop());
        const wavBlob = await toWav(audioBlob);
        await analyzeAudio(wavBlob);
      };

      mediaRecorder.start();
      mediaRecorderRef.current = mediaRecorder;
      setRecording(true);
    } catch (err) {
      setError('Microphone access denied or unavailable.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop();
      setRecording(false);
    }
  };

  const analyzeAudio = async (audioBlob: Blob) => {
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.wav');

      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Analysis failed.');
      }

      const data = (await response.json()) as AnalysisResult;
      setResults(data);
    } catch (err) {
      setError('Unable to analyze audio.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="analyzer">
      <div className="controls">
        <button
          className={recording ? 'primary stop' : 'primary'}
          onClick={recording ? stopRecording : startRecording}
        >
          {recording ? 'Stop Recording' : 'Start Recording'}
        </button>
        <p className="note">Record 10–30 seconds for best results.</p>
      </div>

      {loading && <p className="status">Analyzing audio…</p>}
      {error && <p className="status error">{error}</p>}

      {results && (
        <div className="results">
          <div className="result-card">
            <h2>Risk level</h2>
            <div className={`risk ${results.riskLevel}`}>
              {results.riskLevel.toUpperCase()} · {(results.deceptionScore * 100).toFixed(1)}%
            </div>
            <p>{results.explanation}</p>
          </div>

          <div className="result-card">
            <h2>Emotional state</h2>
            <p className="dominant">Primary: {results.dominantEmotion}</p>
            <div className="bars">
              {Object.entries(results.emotionalState).map(([emotion, score]) => (
                <div key={emotion} className="bar-row">
                  <span>{emotion}</span>
                  <div className="bar">
                    <div className="fill" style={{ width: `${score * 100}%` }} />
                  </div>
                  <span>{(score * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>

          {results.transcription && (
            <div className="result-card">
              <h2>Transcription</h2>
              <p className="mono">{results.transcription}</p>
            </div>
          )}

          {results.deceptionFactors && results.audioFeatures && (
            <div className="result-card">
              <h2>Acoustic indicators</h2>
              <div className="metrics">
                <div>
                  <strong>Stress score:</strong> {results.audioFeatures.stressIndicators.stressScore.toFixed(2)}
                </div>
                <div>
                  <strong>Pause frequency:</strong> {results.audioFeatures.pauseDetection.pauseFrequency.toFixed(2)} / sec
                </div>
                <div>
                  <strong>Pitch range:</strong> {results.audioFeatures.pitchAnalysis.f0Range.toFixed(0)} Hz
                </div>
                <div>
                  <strong>Speech rate:</strong> {results.audioFeatures.pauseDetection.speechRate.toFixed(0)} WPM
                </div>
                <div>
                  <strong>Formants:</strong> F1 {results.audioFeatures.formants.f1.toFixed(0)} Hz, F2 {results.audioFeatures.formants.f2.toFixed(0)} Hz
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}

async function toWav(audioBlob: Blob): Promise<Blob> {
  const arrayBuffer = await audioBlob.arrayBuffer();
  const audioContext = new AudioContext();
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
  const wavBuffer = encodeWav(audioBuffer);
  audioContext.close();
  return new Blob([wavBuffer], { type: 'audio/wav' });
}

function encodeWav(audioBuffer: AudioBuffer): ArrayBuffer {
  const numChannels = audioBuffer.numberOfChannels;
  const sampleRate = audioBuffer.sampleRate;
  const samples = interleave(audioBuffer);
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * 2, true);
  view.setUint16(32, numChannels * 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  return buffer;
}

function interleave(audioBuffer: AudioBuffer): Float32Array {
  const numChannels = audioBuffer.numberOfChannels;
  if (numChannels === 1) return audioBuffer.getChannelData(0);

  const length = audioBuffer.length * numChannels;
  const result = new Float32Array(length);
  const channels = Array.from({ length: numChannels }, (_, i) => audioBuffer.getChannelData(i));

  let index = 0;
  for (let i = 0; i < audioBuffer.length; i++) {
    for (let ch = 0; ch < numChannels; ch++) {
      result[index++] = channels[ch][i];
    }
  }

  return result;
}

function writeString(view: DataView, offset: number, value: string) {
  for (let i = 0; i < value.length; i++) {
    view.setUint8(offset + i, value.charCodeAt(i));
  }
}
