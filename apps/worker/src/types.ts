export interface AudioFeatures {
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
}

export interface DeceptionAnalysis {
  audioFeatures: AudioFeatures;
  emotionalState: Record<string, number>;
  dominantEmotion: string;
  deceptionScore: number;
  deceptionFactors: {
    stressLevel: number;
    hesitationPattern: number;
    pitchVariation: number;
    speechRate: number;
    pauseFrequency: number;
    formantVariability: number;
  };
  riskLevel: 'low' | 'medium' | 'high';
  explanation: string;
  transcription?: string;
}

export interface Env {
  ACCOUNT_ID: string;
  GATEWAY_ID: string;
  HF_API_TOKEN: string;
}
