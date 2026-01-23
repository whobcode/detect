import { Router } from 'itty-router';
import { AudioAnalyzer } from './audio-analyzer';
import type { DeceptionAnalysis, Env } from './types';

const router = Router();
const analyzer = new AudioAnalyzer();

router.post('/api/analyze', async (request: Request, env: Env) => {
  try {
    const formData = await request.formData();
    const audioFile = formData.get('audio');

    if (!(audioFile instanceof File)) {
      return json({ error: 'No audio file provided.' }, 400);
    }

    const arrayBuffer = await audioFile.arrayBuffer();
    const { samples, sampleRate } = analyzer.decodeWav(arrayBuffer);
    const audioFeatures = analyzer.extractFeatures(samples, sampleRate);

    const base64Audio = toBase64(new Uint8Array(arrayBuffer));

    const emotion = await callHuggingFace(
      env,
      'firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3',
      base64Audio
    );

    const transcription = await callHuggingFace(
      env,
      'openai/whisper-large-v3',
      base64Audio
    );

    const { emotionScores, dominantEmotion } = normalizeEmotion(emotion);
    const deceptionFactors = calculateDeceptionFactors(audioFeatures);
    const deceptionScore = computeDeceptionScore(deceptionFactors);
    const riskLevel = getRiskLevel(deceptionScore);

    const analysis: DeceptionAnalysis = {
      audioFeatures,
      emotionalState: emotionScores,
      dominantEmotion,
      deceptionScore,
      deceptionFactors,
      riskLevel,
      explanation: generateExplanation(deceptionScore, deceptionFactors, audioFeatures),
      transcription: typeof transcription?.text === 'string' ? transcription.text : undefined,
    };

    return json(analysis);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error.';
    return json({ error: message }, 500);
  }
});

router.get('/health', () => new Response('OK'));

export default router;

async function callHuggingFace(env: Env, model: string, base64Audio: string) {
  const url = `https://gateway.ai.cloudflare.com/v1/${env.ACCOUNT_ID}/${env.GATEWAY_ID}/huggingface/${model}`;
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${env.HF_API_TOKEN}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ inputs: base64Audio }),
  });

  if (!response.ok) {
    throw new Error(`Hugging Face error: ${response.status} ${response.statusText}`);
  }

  return response.json();
}

function normalizeEmotion(raw: unknown): { emotionScores: Record<string, number>; dominantEmotion: string } {
  const scores: Record<string, number> = {};
  let dominantEmotion = 'neutral';
  let maxScore = 0;

  if (Array.isArray(raw)) {
    for (const item of raw) {
      if (!item || typeof item !== 'object') continue;
      const label = (item as { label?: string }).label;
      const score = (item as { score?: number }).score;
      if (typeof label === 'string' && typeof score === 'number') {
        scores[label] = score;
        if (score > maxScore) {
          maxScore = score;
          dominantEmotion = label;
        }
      }
    }
  }

  return { emotionScores: scores, dominantEmotion };
}

function calculateDeceptionFactors(audioFeatures: DeceptionAnalysis['audioFeatures']): DeceptionAnalysis['deceptionFactors'] {
  return {
    stressLevel: audioFeatures.stressIndicators.stressScore,
    hesitationPattern: Math.min(audioFeatures.pauseDetection.pauseFrequency * 0.5, 1),
    pitchVariation: Math.min(audioFeatures.pitchAnalysis.f0Range / 200, 1),
    speechRate: Math.abs(audioFeatures.pauseDetection.speechRate - 150) / 150,
    pauseFrequency: Math.min(audioFeatures.pauseDetection.pauseFrequency * 2, 1),
    formantVariability:
      (Math.abs(audioFeatures.formants.f1 - 700) + Math.abs(audioFeatures.formants.f2 - 1200)) / 2000,
  };
}

function computeDeceptionScore(factors: DeceptionAnalysis['deceptionFactors']): number {
  const weights = {
    stressLevel: 0.25,
    hesitationPattern: 0.2,
    pitchVariation: 0.15,
    speechRate: 0.15,
    pauseFrequency: 0.15,
    formantVariability: 0.1,
  };

  const score =
    factors.stressLevel * weights.stressLevel +
    factors.hesitationPattern * weights.hesitationPattern +
    factors.pitchVariation * weights.pitchVariation +
    factors.speechRate * weights.speechRate +
    factors.pauseFrequency * weights.pauseFrequency +
    factors.formantVariability * weights.formantVariability;

  return Math.min(Math.max(score, 0), 1);
}

function getRiskLevel(score: number): 'low' | 'medium' | 'high' {
  if (score < 0.33) return 'low';
  if (score < 0.67) return 'medium';
  return 'high';
}

function generateExplanation(
  score: number,
  factors: DeceptionAnalysis['deceptionFactors'],
  features: DeceptionAnalysis['audioFeatures']
): string {
  const parts: string[] = [];

  if (factors.stressLevel > 0.6) {
    parts.push(
      `High vocal stress detected (jitter ${(features.stressIndicators.jitter * 100).toFixed(1)}%, shimmer ${(features.stressIndicators.shimmer * 100).toFixed(1)}%).`
    );
  }

  if (factors.pauseFrequency > 0.6) {
    parts.push(
      `Frequent pauses (${features.pauseDetection.pauseCount} pauses, avg ${features.pauseDetection.meanPauseDuration.toFixed(0)}ms).`
    );
  }

  if (factors.pitchVariation > 0.6) {
    parts.push(`Significant pitch variation (range ${features.pitchAnalysis.f0Range.toFixed(0)}Hz).`);
  }

  if (score > 0.65) {
    parts.push('Multiple indicators present — high suspicion.');
  } else if (score > 0.4) {
    parts.push('Some indicators present — moderate suspicion.');
  } else {
    parts.push('Minimal indicators detected — low suspicion.');
  }

  return parts.join(' ');
}

function toBase64(bytes: Uint8Array): string {
  let binary = '';
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}
