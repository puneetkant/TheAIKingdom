"""
Working Example 2: Text-to-Audio / Speech
Mel spectrogram computation on a synthetic sine wave,
formant frequency analysis.
Run: python working_example2.py
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


def mel_filterbank(n_filters, n_fft, sr, f_min=0, f_max=None):
    """Create triangular mel filterbank."""
    if f_max is None:
        f_max = sr / 2
    # Convert Hz to Mel
    mel_min = 2595 * np.log10(1 + f_min / 700)
    mel_max = 2595 * np.log10(1 + f_max / 700)
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filters = np.zeros((n_filters, n_fft // 2 + 1))
    for m in range(1, n_filters + 1):
        f_left, f_center, f_right = bin_points[m-1], bin_points[m], bin_points[m+1]
        for k in range(f_left, f_center):
            filters[m-1, k] = (k - f_left) / (f_center - f_left + 1e-10)
        for k in range(f_center, min(f_right, n_fft // 2 + 1)):
            filters[m-1, k] = (f_right - k) / (f_right - f_center + 1e-10)
    return filters


def mel_spectrogram(signal, sr=16000, n_fft=256, hop_length=128, n_mels=40):
    """Compute mel spectrogram using STFT."""
    # STFT
    window = np.hanning(n_fft)
    n_frames = (len(signal) - n_fft) // hop_length + 1
    stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=complex)
    for i in range(n_frames):
        frame = signal[i * hop_length: i * hop_length + n_fft] * window
        stft[:, i] = np.fft.rfft(frame)
    power = np.abs(stft) ** 2

    # Apply mel filterbank
    fb = mel_filterbank(n_mels, n_fft, sr)
    mel_spec = fb @ power
    return np.log(mel_spec + 1e-10)  # log-mel


def demo():
    print("=== Text-to-Audio: Mel Spectrogram ===")
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Synthetic vowel-like signal: fundamental + formants
    f0 = 120      # fundamental frequency
    f1 = 800      # first formant (vowel "ah")
    f2 = 1200     # second formant
    signal = (0.6 * np.sin(2 * np.pi * f0 * t) +
              0.3 * np.sin(2 * np.pi * f1 * t) +
              0.1 * np.sin(2 * np.pi * f2 * t))
    # Mild envelope
    envelope = np.exp(-t * 0.5)
    signal = signal * envelope

    print(f"  Signal duration: {duration}s at {sr} Hz")
    print(f"  Fundamental: {f0} Hz, Formants: {f1} Hz, {f2} Hz")

    mel_spec = mel_spectrogram(signal, sr=sr, n_fft=512, hop_length=256, n_mels=40)
    print(f"  Mel spectrogram shape: {mel_spec.shape}")

    # FFT for formant analysis
    fft_mag = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
    freqs = np.fft.rfftfreq(len(signal), 1 / sr)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Waveform
    axes[0][0].plot(t[:500], signal[:500], color="steelblue", lw=1)
    axes[0][0].set(xlabel="Time (s)", ylabel="Amplitude",
                   title="Synthetic Speech Waveform (first 500 samples)")
    axes[0][0].grid(True, alpha=0.3)

    # FFT spectrum
    axes[0][1].plot(freqs[:3000], fft_mag[:3000], color="tomato", lw=1)
    axes[0][1].axvline(f0, color="blue", linestyle="--", alpha=0.7, label=f"f0={f0}Hz")
    axes[0][1].axvline(f1, color="green", linestyle="--", alpha=0.7, label=f"F1={f1}Hz")
    axes[0][1].axvline(f2, color="purple", linestyle="--", alpha=0.7, label=f"F2={f2}Hz")
    axes[0][1].set(xlabel="Frequency (Hz)", ylabel="Magnitude",
                   title="FFT Spectrum (Formant Analysis)")
    axes[0][1].legend(fontsize=8)
    axes[0][1].grid(True, alpha=0.3)

    # Mel spectrogram
    im = axes[1][0].imshow(mel_spec, aspect="auto", origin="lower",
                            cmap="magma", interpolation="nearest")
    axes[1][0].set(xlabel="Time Frame", ylabel="Mel Filter",
                   title="Log-Mel Spectrogram")
    plt.colorbar(im, ax=axes[1][0], label="Log Power")

    # Mel filterbank
    fb = mel_filterbank(40, 512, sr)
    for i in range(0, 40, 5):
        axes[1][1].plot(fb[i], alpha=0.6)
    axes[1][1].set(xlabel="FFT Bin", ylabel="Filter Weight",
                   title="Mel Filterbank (every 5th filter)")
    axes[1][1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / "mel_spectrogram.png", dpi=100)
    plt.close()
    print("  Saved mel_spectrogram.png")


if __name__ == "__main__":
    demo()
