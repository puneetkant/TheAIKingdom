# 6.4.4 Text-to-Audio and Speech

Text-to-speech (TTS) and audio generation models (Whisper, Bark, MusicGen, VoiceBox) convert text or prompts into natural speech or music. Core components include mel-filterbank extraction, vocoder synthesis, and prosody modelling. This folder implements mel filterbanks from scratch, STFT, log-mel spectrograms, and formant frequency analysis.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | STFT, mel filterbank construction, log-mel spectrogram, formant FFT analysis |
| `working_example.ipynb` | Interactive notebook version |
| `output/` | `mel_spectrogram.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| Mel filterbank | Perceptually-scaled frequency bins |
| STFT | Short-Time Fourier Transform; time-frequency representation |
| Vocoder | Converts mel spectrograms to waveforms (HiFi-GAN) |
| Prosody | Pitch, rhythm, and emphasis in speech |
| Codec model | Encodec-style discrete audio tokens |

## Learning Resources

- Radford et al. *Whisper* (2023)
- Shen et al. *Natural TTS Synthesis* / Tacotron (2018)
- Copet et al. *MusicGen* (2023)
