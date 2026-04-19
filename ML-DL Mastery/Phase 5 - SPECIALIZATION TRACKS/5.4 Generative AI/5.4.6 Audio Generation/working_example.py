"""
Working Example: Audio Generation
Covers audio fundamentals, WaveNet, speech synthesis, music generation,
voice cloning, and audio diffusion models.
"""
import numpy as np
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_audio_gen")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. Audio fundamentals -----------------------------------------------------
def audio_fundamentals():
    print("=== Audio Fundamentals ===")
    SR = 22050  # sample rate

    print(f"  Sample rate: {SR} Hz (22.05 kHz)")
    print(f"  1 second of audio = {SR} samples")
    print(f"  CD quality: 44100 Hz, 16-bit stereo")
    print()

    # Synthesise a simple tone
    duration = 0.5; freq = 440  # A4
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t)

    # STFT (Short-Time Fourier Transform)
    frame_len = 1024; hop = 256
    n_frames  = (len(signal) - frame_len) // hop + 1
    window    = np.hanning(frame_len)
    S         = np.zeros((frame_len//2+1, n_frames))
    for i in range(n_frames):
        frame = signal[i*hop : i*hop+frame_len] * window
        fft   = np.fft.rfft(frame)
        S[:, i] = np.abs(fft)

    print(f"  440 Hz tone: {len(signal)} samples")
    print(f"  STFT frames: {n_frames}  freq bins: {S.shape[0]}")
    print(f"  Max energy at bin: {S[:, n_frames//2].argmax()} "
          f"({S[:, n_frames//2].argmax() * SR / frame_len:.1f} Hz)")

    # Mel scale
    def hz_to_mel(hz): return 2595 * np.log10(1 + hz/700)
    def mel_to_hz(mel): return 700 * (10**(mel/2595) - 1)

    print()
    print("  Mel frequency scale:")
    for hz in [100, 500, 1000, 2000, 4000, 8000]:
        print(f"    {hz:>5} Hz -> {hz_to_mel(hz):.1f} mel")

    print()
    print("  Audio representations:")
    reps = [
        ("Waveform",      "Raw samples; high dim; directly audible"),
        ("Spectrogram",   "|STFT|; complex; invertible via Griffin-Lim"),
        ("Mel-spec",      "Mel-filtered spectrogram; perceptually motivated; ~80 bins"),
        ("Log-mel",       "log(mel-spec); normalise energy range; used in most models"),
        ("MFCCs",         "Cosine transform of log-mel; compact; classic ASR feature"),
        ("Discrete tokens","Codec tokens (EnCodec/DAC); VQ; 75Hz; modern models"),
    ]
    for r, d in reps:
        print(f"    {r:<16} {d}")

    # Save spectrogram plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(t[:500], signal[:500])
    axes[0].set_title("440 Hz Sine Wave"); axes[0].set_xlabel("Time (s)")
    axes[1].imshow(np.log1p(S), aspect="auto", origin="lower",
                   extent=[0, duration, 0, SR/2])
    axes[1].set_title("Spectrogram"); axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Hz")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "audio_basics.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"\n  Audio basics plot: {path}")


# -- 2. Text-to-Speech (TTS) architectures ------------------------------------
def tts_architectures():
    print("\n=== Text-to-Speech (TTS) Architectures ===")
    print()
    print("  Classic pipeline:")
    print("    Text -> Text Preprocessing -> Acoustic Model -> Vocoder -> Audio")
    print()
    print("  Acoustic model: text/phonemes -> mel-spectrogram")
    print("  Vocoder: mel-spectrogram -> waveform")
    print()

    models = [
        ("WaveNet",      2016, "Autoregressive; 16kHz; gold standard; slow"),
        ("Tacotron",     2017, "Seq2Seq; attention; mel output -> Griffin-Lim"),
        ("Tacotron 2",   2018, "Tacotron + WaveNet vocoder; natural speech"),
        ("FastSpeech",   2019, "Non-autoregressive; duration predictor; fast"),
        ("FastSpeech 2", 2020, "+ pitch, energy; more expressive"),
        ("VITS",         2021, "VAE-GAN; end-to-end; natural; parallel"),
        ("YourTTS",      2022, "Zero-shot voice cloning; multilingual"),
        ("XTTS",         2023, "Coqui; cross-lingual zero-shot; real-time"),
        ("NaturalSpeech3",2024,"Codec LM + FACodec; near-human quality"),
        ("ElevenLabs",   2024, "Commercial; voice cloning; multilingual"),
    ]
    print(f"  {'Model':<18} {'Year'} {'Notes'}")
    print(f"  {'-'*18} {'-'*4} {'-'*50}")
    for m, y, d in models:
        print(f"  {m:<18} {y}  {d}")


# -- 3. Vocoders ---------------------------------------------------------------
def vocoders():
    print("\n=== Neural Vocoders ===")
    print("  Convert mel-spectrogram -> waveform")
    print()
    vocoders_list = [
        ("WaveNet",      "Autoregressive; 22kHz; 750ms/s on GPU -> slow"),
        ("WaveGlow",     "Normalising flow; fast; good quality; 2019"),
        ("WaveGrad",     "Diffusion vocoder; controllable quality/speed"),
        ("HiFi-GAN",     "GAN-based; very fast; high quality; default for TTS"),
        ("BigVGAN",      "Alias-free; anti-aliased; beats HiFi-GAN"),
        ("Vocos",        "Lightweight; frequency domain; 2023"),
        ("EnCodec",      "Neural codec for streaming; Meta; 24kHz"),
        ("DAC",          "Descript Audio Codec; 44kHz; better compression"),
    ]
    for v, d in vocoders_list:
        print(f"  {v:<12} {d}")

    # Simulate HiFi-GAN discriminator structure
    print()
    print("  HiFi-GAN multi-scale + multi-period discriminators:")
    print("    Multi-scale: 3 discriminators on {1x, 2x, 4x} downsampled audio")
    print("    Multi-period: 5 discriminators at periods {2,3,5,7,11}")
    print("    Feature matching loss + mel spectrogram loss + adversarial loss")


# -- 4. Music generation -------------------------------------------------------
def music_generation():
    print("\n=== Music Generation Models ===")
    models = [
        ("Jukebox",       2020, "OpenAI; VQ-VAE + Transformer; multi-genre; minutes-long"),
        ("MusicLM",       2023, "Google; text-to-music; AudioLM + MuLan"),
        ("MusicGen",      2023, "Meta; text/melody conditioned; 32kHz; open weights"),
        ("AudioCraft",    2023, "Meta suite: MusicGen + AudioGen + EnCodec"),
        ("Suno",          2024, "Commercial; vocals + instruments; text-to-full-song"),
        ("Udio",          2024, "Commercial; competitor to Suno; high quality"),
        ("AudioLDM",      2023, "Latent diffusion for audio; text-to-sound effects"),
        ("Stable Audio",  2024, "Stability AI; 44kHz; stereo; up to 3 min"),
    ]
    print(f"  {'Model':<16} {'Year'} {'Notes'}")
    print(f"  {'-'*16} {'-'*4} {'-'*50}")
    for m, y, d in models:
        print(f"  {m:<16} {y}  {d}")
    print()
    print("  Typical pipeline:")
    print("    Text -> Language model embedding -> Audio token LM -> EnCodec decode -> Waveform")


# -- 5. Voice cloning demo (conceptual) ---------------------------------------
def voice_cloning():
    print("\n=== Voice Cloning ===")
    print("  Goal: synthesise speech in target speaker's voice")
    print("         from just a few seconds of reference audio")
    print()
    print("  Speaker encoding:")
    print("    Reference audio -> speaker embedding (d-vector or x-vector)")
    print("    Speaker embedding conditions acoustic model and vocoder")
    print()
    print("  Zero-shot (from one utterance):")
    print("    3-5 seconds reference -> embedding -> condition TTS")
    print("    YourTTS, XTTS, ElevenLabs, VALL-E")
    print()
    print("  VALL-E (Microsoft 2023):")
    print("    3 second prompt -> in-context learning in codec LM")
    print("    Acoustic tokens at 75 Hz from EnCodec")
    print("    Multi-level codebooks: coarse (AR) + fine (NAR)")
    print()
    print("  Codec LM approach (EnCodec tokens):")
    print("    Encode speech -> discrete tokens (8 codebooks x 75Hz)")
    print("    Train LM on token sequences")
    print("    Condition on speaker embedding + text phonemes")
    print()

    # Simulate speaker embedding
    rng = np.random.default_rng(0)
    n_speakers = 5; embed_dim = 256
    speaker_embeddings = rng.standard_normal((n_speakers, embed_dim))
    # Cosine similarity matrix
    norms = np.linalg.norm(speaker_embeddings, axis=1, keepdims=True)
    spk_norm = speaker_embeddings / norms
    sim = spk_norm @ spk_norm.T

    print(f"  Simulated speaker embeddings: {speaker_embeddings.shape}")
    print(f"  Cosine similarity matrix:")
    for i in range(n_speakers):
        row = " ".join(f"{sim[i,j]:+.2f}" for j in range(n_speakers))
        print(f"    Speaker {i}: {row}")
    print(f"  Self-similarity = 1.00; cross-speaker < 1.00 (good disentanglement)")


if __name__ == "__main__":
    audio_fundamentals()
    tts_architectures()
    vocoders()
    music_generation()
    voice_cloning()
