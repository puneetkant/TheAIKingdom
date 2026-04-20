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

def demo_wavetable_synthesis():
    """Wavetable synthesis: generate multiple waveform types."""
    print("\n=== Wavetable Synthesis ===")
    t = np.linspace(0, 1.0, SR, endpoint=False)
    f0 = 220.0  # A3

    waves = {
        "Sine":     np.sin(2*np.pi*f0*t),
        "Square":   np.sign(np.sin(2*np.pi*f0*t)),
        "Sawtooth": 2 * (t*f0 - np.floor(0.5 + t*f0)),
        "Triangle": 2*np.abs(2*(t*f0 - np.floor(t*f0 + 0.5))) - 1,
    }
    for name, wave in waves.items():
        rms = np.sqrt((wave**2).mean())
        spec = np.abs(np.fft.rfft(wave[:SR//4]))
        peak_freq = np.argmax(spec) * SR / (SR//4)
        print(f"  {name:10s}: RMS={rms:.4f}  peak_freq~{peak_freq:.0f}Hz")


def demo_mel_filterbank():
    """Simulate a mel filterbank and extract mel-frequency energies."""
    print("\n=== Mel Filterbank Features ===")
    freqs = [440, 554, 659]
    chord = sum(make_tone(f, duration=0.5) for f in freqs) / len(freqs)
    spec = stft_magnitude(chord)   # (freq_bins, time_frames)
    n_fft = 512

    # Build simple triangular mel filters (n_mel=8 between 80-4000 Hz)
    n_mel = 8
    f_min, f_max = 80.0, 4000.0
    mel_min = 2595 * np.log10(1 + f_min/700)
    mel_max = 2595 * np.log10(1 + f_max/700)
    mel_pts  = np.linspace(mel_min, mel_max, n_mel+2)
    hz_pts   = 700 * (10**(mel_pts/2595) - 1)
    bin_pts  = np.floor((n_fft+1) * hz_pts / SR).astype(int)
    n_bins   = spec.shape[0]
    mel_energies = []
    for m in range(1, n_mel+1):
        lo, ctr, hi = bin_pts[m-1], bin_pts[m], bin_pts[m+1]
        lo = min(lo, n_bins-1); ctr = min(ctr, n_bins); hi = min(hi, n_bins)
        energy = spec[lo:hi, :].mean()
        mel_energies.append(energy)
    print(f"  Mel energies (8 bands): {[round(e,4) for e in mel_energies]}")
    peak_band = int(np.argmax(mel_energies)) + 1
    print(f"  Dominant mel band: {peak_band}")


if __name__ == "__main__":
    demo()
    demo_wavetable_synthesis()
    demo_mel_filterbank()
