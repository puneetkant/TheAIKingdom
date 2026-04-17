"""
Working Example 2: Audio Generation — waveform synthesis and spectrogram features
==================================================================================
Generates synthetic audio waveforms and computes mel-spectrogram-like features.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

SR = 16000  # sample rate

def make_tone(freq=440.0, duration=0.5, sr=SR, envelope=True):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wave = np.sin(2 * np.pi * freq * t)
    if envelope:  # simple ADSR-like ramp
        env = np.minimum(t / 0.05, 1.0) * np.exp(-3 * t)
        wave *= env / (env.max() + 1e-8)
    return wave

def stft_magnitude(wave, n_fft=512, hop=256):
    """Short-time Fourier transform (magnitude)."""
    window = np.hanning(n_fft)
    frames = []
    for start in range(0, len(wave) - n_fft, hop):
        frame = wave[start:start+n_fft] * window
        frames.append(np.abs(np.fft.rfft(frame)))
    return np.array(frames).T  # (freq_bins, time_frames)

def demo():
    print("=== Audio Generation: Waveform + Spectrogram ===")
    # Synthesise a chord (A4 + C#5 + E5)
    freqs = [440, 554, 659]
    chord = sum(make_tone(f, duration=1.0) for f in freqs) / len(freqs)
    print(f"  Chord waveform: {len(chord)} samples at {SR} Hz")

    spec = stft_magnitude(chord)
    print(f"  Spectrogram shape: {spec.shape}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    t = np.linspace(0, 1.0, len(chord))
    axes[0].plot(t[:400], chord[:400]); axes[0].set_title("Waveform (first 25 ms)")
    axes[0].set_xlabel("Time (s)")
    freq_axis = np.linspace(0, SR//2, spec.shape[0])
    axes[1].imshow(np.log1p(spec[:100]), aspect="auto", origin="lower",
                   extent=[0, 1.0, 0, freq_axis[100]])
    axes[1].set_title("Log-Magnitude Spectrogram (0–1.4 kHz)")
    axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Freq (Hz)")
    plt.tight_layout(); plt.savefig(OUTPUT / "audio_spectrogram.png"); plt.close()

    # Harmonic ratios as "feature vector"
    feat = spec[:, spec.shape[1]//2].copy()
    feat /= feat.max() + 1e-8
    print(f"  Peak frequency bins: {np.argsort(feat)[-3:][::-1]}")
    print("  Saved audio_spectrogram.png")

if __name__ == "__main__":
    demo()
