"""
Working Example: Text-to-Audio and Speech Synthesis
Covers TTS, speech recognition (ASR), music generation,
and sound effect generation.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_audio")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Audio modalities overview ──────────────────────────────────────────────
def audio_overview():
    print("=== Text-to-Audio and Speech ===")
    print()
    modalities = [
        ("TTS (Text-to-Speech)",        "Natural human speech synthesis"),
        ("Voice cloning",               "Replicate target voice from short sample"),
        ("ASR (Speech Recognition)",    "Audio → text transcription"),
        ("Music generation",            "Full songs from text prompts"),
        ("Sound effect generation",     "Environmental/foley sounds from description"),
        ("Speech-to-speech",            "Real-time voice conversion / translation"),
    ]
    for m, d in modalities:
        print(f"  {m:<34} {d}")


# ── 2. TTS systems ────────────────────────────────────────────────────────────
def tts_systems():
    print("\n=== Text-to-Speech Systems ===")
    print()
    systems = [
        ("Tacotron 2",    "2019; text → mel spectrogram; WaveNet vocoder"),
        ("FastSpeech 2",  "Non-autoregressive; duration/pitch/energy predictors"),
        ("VITS",          "End-to-end; VAE + normalising flow; natural prosody"),
        ("NaturalSpeech 2","Latent diffusion; zero-shot voice cloning; codec"),
        ("Voicebox",      "Meta; flow matching; in-context speech editing"),
        ("SeedTTS",       "ByteDance; multilingual; emotion control"),
        ("Kokoro",        "Small (82M); fast; open weights; Apache 2.0"),
        ("Dia",           "Nari Labs; 1.6B; emotive; dialogue"),
        ("ElevenLabs",    "Commercial; voice cloning leader; API"),
        ("OpenAI TTS",    "GPT-4o voice mode; real-time; emotional nuance"),
    ]
    print(f"  {'System':<20} {'Notes'}")
    for s, d in systems:
        print(f"  {s:<20} {d}")
    print()
    print("  TTS pipeline:")
    print("    Text → grapheme-to-phoneme (G2P)")
    print("         → acoustic model (phones → mel spectrogram)")
    print("         → vocoder (mel → waveform)")
    print()
    print("  Modern end-to-end: text → codec tokens → audio (no spectrogram)")


# ── 3. Audio codecs and representations ──────────────────────────────────────
def audio_representations():
    print("\n=== Audio Representations ===")
    print()
    print("  Waveform: raw samples at e.g. 16kHz (16k values/sec)")
    print("  Mel spectrogram: time-frequency; 80-128 mel bins; standard for TTS")
    print("  Neural codecs: compress audio → discrete tokens (8x+ compression)")
    print()
    codecs = [
        ("EnCodec",     "Meta; RVQ codec; 75 tokens/sec at 24kHz"),
        ("Descript DAC","Improved EnCodec; lower bitrate; widely used"),
        ("SoundStream", "Google; streaming; phone quality"),
        ("Mimi",        "Kyutai; low-latency; streaming"),
    ]
    print("  Neural audio codecs (Residual Vector Quantisation):")
    for c, d in codecs:
        print(f"  {c:<16} {d}")
    print()

    # Simulate RVQ quantisation
    rng = np.random.default_rng(0)
    n_codebooks = 4
    codebook_size = 1024
    frame = rng.normal(0, 1, 128)  # 128-dim audio frame
    print("  Simulated RVQ encoding of one audio frame:")
    residual = frame.copy()
    codes = []
    for cb in range(n_codebooks):
        codebook = rng.normal(0, 1, (codebook_size, 128))
        sims = codebook @ residual
        best = np.argmax(sims)
        codes.append(best)
        residual = residual - codebook[best]
        print(f"    Codebook {cb}: code={best:4d}  residual norm={np.linalg.norm(residual):.3f}")
    print(f"  Encoded as {n_codebooks} tokens: {codes}")


# ── 4. Music and sound generation ────────────────────────────────────────────
def music_generation():
    print("\n=== Music and Sound Generation ===")
    print()
    systems = [
        ("MusicGen",    "Meta; mono/stereo; melody conditioning; open"),
        ("MusicLM",     "Google; semantic/acoustic tokens; high quality"),
        ("Stable Audio","Stability AI; latent diffusion; 45s+ songs"),
        ("Suno",        "Vocals + instruments; full songs; commercial"),
        ("Udio",        "Similar to Suno; commercial"),
        ("AudioCraft",  "Meta suite: MusicGen + AudioGen + EnCodec"),
        ("AudioGen",    "Meta; environmental/foley sound effects"),
        ("Tango",       "Text→audio; Foley; open weights; DCASE winner"),
    ]
    print(f"  {'System':<16} {'Notes'}")
    for s, d in systems:
        print(f"  {s:<16} {d}")
    print()
    print("  Music generation architectures:")
    arches = [
        ("Token autoregressive", "Text → codec tokens → audio; MusicGen"),
        ("Diffusion",            "Text → mel/latent → audio; Stable Audio, Tango"),
        ("Dual codec",           "Semantic tokens (coarse) + acoustic (fine)"),
    ]
    for a, d in arches:
        print(f"  {a:<24} {d}")


# ── 5. ASR – Whisper ──────────────────────────────────────────────────────────
def asr_whisper():
    print("\n=== ASR: Whisper Architecture ===")
    print()
    print("  Whisper (OpenAI 2022): encoder-decoder transformer")
    print("  Trained on 680K hours of weakly supervised web audio")
    print()
    print("  Architecture:")
    print("    Audio → 80-channel log-Mel spectrogram (25ms window, 10ms stride)")
    print("    Encoder: Conv1D + Transformer (positional: learned)")
    print("    Decoder: Transformer; predicts text BPE tokens")
    print("    Special tokens: <|startoftranscript|> <|en|> <|transcribe|>")
    print()
    models = [
        ("tiny",   "39M",  "~10x faster than base; low accuracy"),
        ("base",   "74M",  "Decent; fast"),
        ("small",  "244M", "Good balance"),
        ("medium", "769M", "High accuracy"),
        ("large-v3","1.5B","Best; multilingual; 1500h per lang"),
    ]
    print(f"  {'Size':<10} {'Params':<8} {'Notes'}")
    for s, p, d in models:
        print(f"  {s:<10} {p:<8} {d}")
    print()
    print("  Faster-Whisper: CTranslate2 backend; 4x faster with same accuracy")
    print("  WhisperX: word-level timestamps; speaker diarisation")


if __name__ == "__main__":
    audio_overview()
    tts_systems()
    audio_representations()
    music_generation()
    asr_whisper()
