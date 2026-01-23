# Deception Indicator Analyzer (Audio)

## Important caveat
True "lie detection" via audio is pseudoscience. This system does **not** detect lies. It surfaces **deception indicators** by analyzing:

- Voice stress/tension indicators
- Emotional tone and sentiment
- Speech patterns and hesitation
- Voice characteristics that can correlate with deception in specific contexts (not conclusive)

Use this as a decision aid, not as proof of truthfulness.

---

## Architecture (Cloudflare Pages + Workers + Hugging Face)

Browser (audio capture)
→ Cloudflare Pages (TypeScript frontend)
→ Cloudflare Worker (TypeScript)
→ Hugging Face Inference API (via Cloudflare AI Gateway)
→ Analysis results back to frontend

---

## What’s implemented in this design
The analyzer includes **acoustic feature extraction** in addition to model inference:

- Speaker stress analysis (jitter, shimmer, NHR)
- Pause detection (frequency and duration)
- Vocal pitch analysis (F0, range, vibrato)
- Formant extraction (F1–F3)
- Spectral features (MFCCs, centroid, rolloff, ZCR)

---

## Recommended Hugging Face models

- **Speech-to-Text:** `openai/whisper-large-v3`
- **Emotion/Stress (audio classification):** pick a strong speech-emotion-recognition model
- **Sentiment (text):** any robust text sentiment model

Model choices should be validated against your use case and dataset.

---

## Worker (TypeScript) — high-level flow

1. Receive audio file from frontend
2. Decode audio and extract acoustic features
3. Call HF models (emotion + optional transcription)
4. Compute a composite “deception indicator score” (heuristic)
5. Return structured analysis to the frontend

---

## Frontend (TypeScript + React)

- Record microphone audio in browser
- POST to `/api/analyze`
- Render:
  - emotional state distribution
  - deception indicator score
  - acoustic feature breakdown
  - plain-English analysis summary

---

## Fine-tuning pipeline (custom datasets)

### Goal
Train a **binary classifier** for “truth vs deception” based on labeled audio samples, or train an **emotion/stress classifier** to better fit your domain.

### Dataset format
Organize labeled data:

```
/data
  /truth
    speaker1_statement1.wav
  /deception
    speaker2_lie1.wav
```

Convert to a JSON/CSV/HF dataset and train with `transformers` (Wav2Vec2, HuBERT, or Whisper-based classifier).

### Suggested training steps

1. Prepare labeled dataset
2. Convert to HF dataset format
3. Fine-tune model (`Wav2Vec2ForSequenceClassification`)
4. Evaluate (accuracy / precision / recall / F1)
5. Push to Hugging Face Hub
6. Update Worker to call your fine-tuned model via AI Gateway

---

## Ethical and product caveats

- This is **not** lie detection.
- Context matters: anxiety, culture, and environment strongly influence speech.
- Always disclose what’s being measured.
- Avoid using for high-stakes decisions without human oversight.

---

## Next steps

- Confirm the exact HF models you want to use
- Add a small labeled dataset (even 50–200 samples) to calibrate heuristics
- Create evaluation benchmarks for your specific domain

