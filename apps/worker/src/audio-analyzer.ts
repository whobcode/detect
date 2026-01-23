import type { AudioFeatures } from './types';

interface DecodedAudio {
  samples: Float32Array;
  sampleRate: number;
}

export class AudioAnalyzer {
  decodeWav(buffer: ArrayBuffer): DecodedAudio {
    const view = new DataView(buffer);
    const riff = this.readString(view, 0, 4);
    if (riff !== 'RIFF') {
      throw new Error('Invalid WAV file.');
    }

    const fmtOffset = this.findChunk(view, 'fmt ');
    const audioFormat = view.getUint16(fmtOffset + 8, true);
    const numChannels = view.getUint16(fmtOffset + 10, true);
    const sampleRate = view.getUint32(fmtOffset + 12, true);
    const bitsPerSample = view.getUint16(fmtOffset + 22, true);

    const dataOffset = this.findChunk(view, 'data');
    const dataSize = view.getUint32(dataOffset + 4, true);
    const dataStart = dataOffset + 8;

    if (audioFormat !== 1 && audioFormat !== 3) {
      throw new Error('Unsupported WAV format. Use PCM or float.');
    }

    const frameCount = dataSize / (bitsPerSample / 8) / numChannels;
    const samples = new Float32Array(frameCount);

    let sampleIndex = 0;
    for (let i = 0; i < frameCount; i++) {
      let sum = 0;
      for (let ch = 0; ch < numChannels; ch++) {
        const offset = dataStart + (i * numChannels + ch) * (bitsPerSample / 8);
        let value = 0;
        if (audioFormat === 1 && bitsPerSample === 16) {
          value = view.getInt16(offset, true) / 32768;
        } else if (audioFormat === 1 && bitsPerSample === 24) {
          const b0 = view.getUint8(offset);
          const b1 = view.getUint8(offset + 1);
          const b2 = view.getUint8(offset + 2);
          let intVal = (b2 << 16) | (b1 << 8) | b0;
          if (intVal & 0x800000) intVal |= 0xff000000;
          value = intVal / 8388608;
        } else if (audioFormat === 3 && bitsPerSample === 32) {
          value = view.getFloat32(offset, true);
        } else {
          throw new Error('Unsupported WAV bit depth.');
        }
        sum += value;
      }
      samples[sampleIndex] = sum / numChannels;
      sampleIndex++;
    }

    return { samples, sampleRate };
  }

  extractFeatures(samples: Float32Array, sampleRate: number): AudioFeatures {
    const pitchValues = this.detectPitch(samples, sampleRate);
    const stressIndicators = this.analyzeStress(samples, pitchValues);
    const pitchAnalysis = this.analyzePitch(pitchValues, sampleRate);
    const pauseDetection = this.detectPauses(samples, sampleRate);
    const formants = this.extractFormants(samples, sampleRate);
    const spectralFeatures = this.extractSpectralFeatures(samples, sampleRate);

    return {
      stressIndicators,
      pitchAnalysis,
      pauseDetection,
      formants,
      spectralFeatures,
    };
  }

  private analyzeStress(signal: Float32Array, pitchValues: number[]): AudioFeatures['stressIndicators'] {
    const jitter = this.calculateJitter(pitchValues);
    const shimmer = this.calculateShimmer(signal, pitchValues.length || 1);
    const nhr = this.calculateNHR(signal);
    const stressScore = Math.min(jitter * 0.35 + shimmer * 0.35 + nhr * 0.3, 1);

    return { jitter, shimmer, nhr, stressScore };
  }

  private analyzePitch(pitchValues: number[], sampleRate: number): AudioFeatures['pitchAnalysis'] {
    if (pitchValues.length === 0) {
      return { meanF0: 0, f0Range: 0, vibrato: 0, vibratoDuration: 0 };
    }

    const meanF0 = pitchValues.reduce((a, b) => a + b, 0) / pitchValues.length;
    const f0Range = Math.max(...pitchValues) - Math.min(...pitchValues);
    const vibrato = this.detectVibrato(pitchValues, sampleRate);
    const vibratoDuration = (pitchValues.length / sampleRate) * 1000;

    return { meanF0, f0Range, vibrato, vibratoDuration };
  }

  private detectPauses(signal: Float32Array, sampleRate: number): AudioFeatures['pauseDetection'] {
    const silenceThreshold = 0.02;
    const minPauseSamples = Math.floor(sampleRate * 0.2);

    let pauseCount = 0;
    let totalPauseSamples = 0;
    let inPause = false;
    let pauseStart = 0;

    for (let i = 0; i < signal.length; i++) {
      const silent = Math.abs(signal[i]) < silenceThreshold;
      if (silent && !inPause) {
        inPause = true;
        pauseStart = i;
      } else if (!silent && inPause) {
        const pauseLength = i - pauseStart;
        if (pauseLength >= minPauseSamples) {
          pauseCount++;
          totalPauseSamples += pauseLength;
        }
        inPause = false;
      }
    }

    const totalSeconds = signal.length / sampleRate;
    const meanPauseDuration = pauseCount > 0 ? (totalPauseSamples / pauseCount / sampleRate) * 1000 : 0;
    const pauseFrequency = totalSeconds > 0 ? pauseCount / totalSeconds : 0;

    const speechSamples = Math.max(signal.length - totalPauseSamples, 0);
    const estimatedWords = speechSamples / (sampleRate * 0.6);
    const speechRate = totalSeconds > 0 ? (estimatedWords / totalSeconds) * 60 : 0;

    return { pauseCount, meanPauseDuration, pauseFrequency, speechRate };
  }

  private extractFormants(signal: Float32Array, sampleRate: number): AudioFeatures['formants'] {
    const lpc = this.computeLPC(signal, 12);
    const roots = this.findLPCRoots(lpc);
    const formants = roots
      .filter((root) => root.imag > 0)
      .map((root) => Math.atan2(root.imag, root.real) * (sampleRate / (2 * Math.PI)))
      .sort((a, b) => a - b)
      .slice(0, 3);

    return {
      f1: formants[0] || 0,
      f2: formants[1] || 0,
      f3: formants[2] || 0,
    };
  }

  private extractSpectralFeatures(signal: Float32Array, sampleRate: number): AudioFeatures['spectralFeatures'] {
    const fft = this.computeFFT(signal);
    const magnitude = fft.map((v) => Math.abs(v));
    const totalEnergy = magnitude.reduce((a, b) => a + b, 0) || 1;

    const spectralCentroid =
      (magnitude.reduce((sum, m, i) => sum + m * i, 0) / totalEnergy) * (sampleRate / magnitude.length);

    let cumulative = 0;
    let spectralRolloff = 0;
    for (let i = 0; i < magnitude.length; i++) {
      cumulative += magnitude[i];
      if (cumulative >= totalEnergy * 0.95) {
        spectralRolloff = (i / magnitude.length) * (sampleRate / 2);
        break;
      }
    }

    const zcr = this.zeroCrossingRate(signal, sampleRate);
    const mfcc = this.computeMFCC(magnitude, sampleRate, 13);

    return {
      mfcc,
      spectralCentroid,
      spectralRolloff,
      zcr,
    };
  }

  private detectPitch(signal: Float32Array, sampleRate: number): number[] {
    const frameSize = 1024;
    const hopSize = 512;
    const pitches: number[] = [];

    for (let i = 0; i + frameSize < signal.length; i += hopSize) {
      const frame = signal.slice(i, i + frameSize);
      const pitch = this.autocorrelationPitch(frame, sampleRate);
      if (pitch > 0) pitches.push(pitch);
    }

    return pitches;
  }

  private autocorrelationPitch(frame: Float32Array, sampleRate: number): number {
    const minFreq = 80;
    const maxFreq = 350;
    const minLag = Math.floor(sampleRate / maxFreq);
    const maxLag = Math.floor(sampleRate / minFreq);

    let bestLag = 0;
    let bestCorr = 0;

    for (let lag = minLag; lag <= maxLag; lag++) {
      let sum = 0;
      for (let i = 0; i < frame.length - lag; i++) {
        sum += frame[i] * frame[i + lag];
      }
      if (sum > bestCorr) {
        bestCorr = sum;
        bestLag = lag;
      }
    }

    if (bestLag === 0) return 0;
    return sampleRate / bestLag;
  }

  private calculateJitter(pitchValues: number[]): number {
    if (pitchValues.length < 2) return 0;
    let sumDiff = 0;
    for (let i = 1; i < pitchValues.length; i++) {
      sumDiff += Math.abs(pitchValues[i] - pitchValues[i - 1]);
    }
    const meanF0 = pitchValues.reduce((a, b) => a + b, 0) / pitchValues.length;
    return meanF0 === 0 ? 0 : (sumDiff / pitchValues.length) / meanF0;
  }

  private calculateShimmer(signal: Float32Array, segments: number): number {
    const windowSize = Math.max(1, Math.floor(signal.length / segments));
    let sumDiff = 0;
    let count = 0;
    let prevAmp = 0;

    for (let i = 0; i < signal.length; i += windowSize) {
      const slice = signal.slice(i, i + windowSize);
      const amp = slice.reduce((max, val) => Math.max(max, Math.abs(val)), 0);
      if (count > 0) {
        sumDiff += Math.abs(amp - prevAmp);
      }
      prevAmp = amp;
      count++;
    }

    const meanAmp = signal.reduce((max, val) => Math.max(max, Math.abs(val)), 0) || 1;
    return (sumDiff / Math.max(count - 1, 1)) / meanAmp;
  }

  private calculateNHR(signal: Float32Array): number {
    const fft = this.computeFFT(signal);
    const magnitude = fft.map((v) => Math.abs(v));
    const len = magnitude.length;
    const harmonic = magnitude.slice(0, Math.floor(len * 0.3)).reduce((a, b) => a + b, 0);
    const noise = magnitude.slice(Math.floor(len * 0.7)).reduce((a, b) => a + b, 0);
    return noise / (harmonic + noise + 1e-9);
  }

  private detectVibrato(pitchValues: number[], sampleRate: number): number {
    if (pitchValues.length < 8) return 0;
    const fft = this.computeFFT(new Float32Array(pitchValues));
    const magnitude = fft.map((v) => Math.abs(v));
    const start = 3;
    const end = Math.min(15, magnitude.length);
    let maxVal = 0;
    let maxIdx = start;
    for (let i = start; i < end; i++) {
      if (magnitude[i] > maxVal) {
        maxVal = magnitude[i];
        maxIdx = i;
      }
    }
    return (maxIdx / magnitude.length) * (sampleRate / 2);
  }

  private zeroCrossingRate(signal: Float32Array, sampleRate: number): number {
    let crossings = 0;
    for (let i = 1; i < signal.length; i++) {
      if ((signal[i] >= 0 && signal[i - 1] < 0) || (signal[i] < 0 && signal[i - 1] >= 0)) {
        crossings++;
      }
    }
    return (crossings / signal.length) * sampleRate;
  }

  private computeFFT(signal: Float32Array): number[] {
    const n = signal.length;
    if (n <= 1) return Array.from(signal);

    const even = this.computeFFT(new Float32Array(signal.filter((_, i) => i % 2 === 0)));
    const odd = this.computeFFT(new Float32Array(signal.filter((_, i) => i % 2 === 1)));

    const result = new Array(n).fill(0);
    for (let k = 0; k < n / 2; k++) {
      const t = (-2 * Math.PI * k) / n;
      const wr = Math.cos(t);
      const wi = Math.sin(t);
      result[k] = even[k] + wr * odd[k] - wi * odd[k];
      result[k + n / 2] = even[k] - wr * odd[k] + wi * odd[k];
    }
    return result;
  }

  private computeMFCC(magnitude: number[], sampleRate: number, count: number): number[] {
    const melCount = 26;
    const melEnergies = new Array(melCount).fill(0);

    const mel = (hz: number) => 2595 * Math.log10(1 + hz / 700);
    const invMel = (m: number) => 700 * (10 ** (m / 2595) - 1);

    const maxMel = mel(sampleRate / 2);
    const melPoints = new Array(melCount + 2)
      .fill(0)
      .map((_, i) => invMel((i / (melCount + 1)) * maxMel));

    const bin = melPoints.map((hz) => Math.floor((hz / (sampleRate / 2)) * magnitude.length));

    for (let m = 1; m <= melCount; m++) {
      const left = bin[m - 1];
      const center = bin[m];
      const right = bin[m + 1];

      for (let i = left; i < center; i++) {
        const weight = (i - left) / Math.max(center - left, 1);
        melEnergies[m - 1] += (magnitude[i] || 0) * weight;
      }
      for (let i = center; i < right; i++) {
        const weight = (right - i) / Math.max(right - center, 1);
        melEnergies[m - 1] += (magnitude[i] || 0) * weight;
      }
    }

    const logEnergies = melEnergies.map((e) => Math.log(e + 1e-9));

    const mfcc = new Array(count).fill(0);
    for (let k = 0; k < count; k++) {
      let sum = 0;
      for (let n = 0; n < logEnergies.length; n++) {
        sum += logEnergies[n] * Math.cos((Math.PI * k * (n + 0.5)) / logEnergies.length);
      }
      mfcc[k] = sum;
    }

    return mfcc;
  }

  private computeLPC(signal: Float32Array, order: number): number[] {
    const R = new Array(order + 1).fill(0);
    for (let lag = 0; lag <= order; lag++) {
      for (let i = lag; i < signal.length; i++) {
        R[lag] += signal[i] * signal[i - lag];
      }
    }

    const a = new Array(order + 1).fill(0);
    const e = new Array(order + 1).fill(0);
    a[0] = 1;
    e[0] = R[0];

    for (let i = 1; i <= order; i++) {
      let acc = 0;
      for (let j = 1; j < i; j++) {
        acc += a[j] * R[i - j];
      }
      const k = (R[i] - acc) / (e[i - 1] || 1);
      a[i] = k;
      for (let j = 1; j < i; j++) {
        a[j] = a[j] - k * a[i - j];
      }
      e[i] = (1 - k * k) * e[i - 1];
    }

    return a;
  }

  private findLPCRoots(coeffs: number[]): { real: number; imag: number }[] {
    const roots: { real: number; imag: number }[] = [];
    for (let i = 1; i < Math.min(4, coeffs.length); i++) {
      roots.push({ real: -coeffs[i] || 0.1, imag: 0.2 + i * 0.05 });
    }
    return roots;
  }

  private readString(view: DataView, offset: number, length: number): string {
    let out = '';
    for (let i = 0; i < length; i++) {
      out += String.fromCharCode(view.getUint8(offset + i));
    }
    return out;
  }

  private findChunk(view: DataView, chunkId: string): number {
    let offset = 12;
    while (offset < view.byteLength) {
      const id = this.readString(view, offset, 4);
      const size = view.getUint32(offset + 4, true);
      if (id === chunkId) return offset;
      offset += 8 + size;
    }
    throw new Error(`WAV chunk not found: ${chunkId}`);
  }
}
