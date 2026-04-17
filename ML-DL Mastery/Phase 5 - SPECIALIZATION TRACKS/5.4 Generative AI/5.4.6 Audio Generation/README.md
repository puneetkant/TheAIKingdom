# 5.4.6 Audio Generation

WaveNet, SoundStream, VALL-E, MusicLM, mel-spectrograms, audio tokenization.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Mel-spectrogram from librosa |
| `working_example2.py` | Synthetic chord waveform + STFT spectrogram (numpy-only) |
| `working_example.ipynb` | Interactive: waveform synthesis + log spectrogram |

## Quick Reference

```python
import librosa, numpy as np

# Load audio
wave, sr = librosa.load("audio.wav", sr=16000)

# Mel-spectrogram
mel = librosa.feature.melspectrogram(y=wave, sr=sr, n_mels=80, n_fft=1024, hop_length=256)
log_mel = librosa.power_to_db(mel)   # (80, T)

# STFT (manual)
n_fft = 512; hop = 256; window = np.hanning(n_fft)
spec = np.array([np.abs(np.fft.rfft(wave[i:i+n_fft]*window))
                 for i in range(0, len(wave)-n_fft, hop)]).T

# Text-to-speech pipeline (HuggingFace)
from transformers import pipeline
tts = pipeline("text-to-audio", model="suno/bark-small")
output = tts("Hello world")
```

## Audio Generation Models

| Model | Type | Output |
|-------|------|--------|
| WaveNet | Autoregressive | Raw waveform |
| SoundStream | Neural codec | Compressed audio |
| VALL-E | Autoregressive + codec | TTS |
| MusicLM | Cascaded diffusion | Music |
| AudioCraft | Autoregressive | Music + SFX |

## Learning Resources
- [WaveNet paper](https://arxiv.org/abs/1609.03499)
- [AudioCraft (Meta)](https://github.com/facebookresearch/audiocraft)

Explore this topic with a small practical project or coding exercise.

## What to build

- Try a small hands-on exercise focused on this topic.
- Keep the code in `project.py` in this folder.
- Add notes, examples, or results inside this directory.

## Suggestions

1. Read the checklist topic and identify one practice task.
2. Write code in `project.py` that illustrates the main concept.
3. Run your code and iterate until it works.

## Notes

- Use Python and standard libraries when possible.
- For data topics, install `numpy`, `pandas`, `matplotlib` as needed.
