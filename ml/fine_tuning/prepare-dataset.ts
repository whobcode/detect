import fs from 'node:fs';
import path from 'node:path';

interface TrainingExample {
  audio_path: string;
  label: 0 | 1;
  speaker_id: string;
}

function collectExamples(dir: string, label: 0 | 1): TrainingExample[] {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir).map((file) => ({
    audio_path: path.join(dir, file),
    label,
    speaker_id: file.split('_')[0] || 'unknown',
  }));
}

function main() {
  const inputDir = process.argv[2] || './data';
  const outputFile = process.argv[3] || './dataset.json';

  const truthDir = path.join(inputDir, 'truth');
  const deceptionDir = path.join(inputDir, 'deception');

  const examples = [
    ...collectExamples(truthDir, 0),
    ...collectExamples(deceptionDir, 1),
  ];

  const dataset = {
    version: '1.0.0',
    data: examples.map((item, idx) => ({
      id: idx,
      audio: {
        path: item.audio_path,
        array: null,
        sampling_rate: 16000,
      },
      label: item.label,
      speaker_id: item.speaker_id,
    })),
  };

  fs.writeFileSync(outputFile, JSON.stringify(dataset, null, 2));
  console.log(`Wrote ${examples.length} examples to ${outputFile}`);
}

main();
