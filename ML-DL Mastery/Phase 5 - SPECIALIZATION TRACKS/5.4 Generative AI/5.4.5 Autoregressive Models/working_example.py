"""
Working Example: Autoregressive Generative Models
Covers autoregressive language models, PixelCNN, WaveNet,
next-token prediction, beam search, and sampling strategies.
"""
import numpy as np
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_autoregressive")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


# -- 1. Autoregressive modelling principle -------------------------------------
def autoregressive_principle():
    print("=== Autoregressive Generative Models ===")
    print()
    print("  Factorise joint distribution via chain rule:")
    print("    p(x_1, ..., x_T) = Pi_{t=1}^T p(x_t | x_1, ..., x_{t-1})")
    print()
    print("  Training: teacher forcing — predict next token given context")
    print("    L = -Sigma_t log p_theta(x_t | x_{<t})")
    print()
    print("  Generation: sample autoregressively")
    print("    x_1 ~ p(x_1)")
    print("    x_2 ~ p(x_2 | x_1)")
    print("    ...")
    print()
    print("  Examples:")
    examples = [
        ("Language",    "GPT, LLaMA: tokens = BPE subwords"),
        ("Image",       "PixelRNN, PixelCNN: pixels in raster order"),
        ("Audio",       "WaveNet: raw audio samples (16kHz)"),
        ("Music",       "MusicGen: mel-spectrogram tokens"),
        ("Video",       "CogVideo: spatial-temporal tokens"),
        ("Protein",     "ESMFold: amino acid sequence"),
    ]
    for dom, ex in examples:
        print(f"  {dom:<10} {ex}")


# -- 2. N-gram language model (numpy) -----------------------------------------
def ngram_lm_demo():
    print("\n=== N-Gram Language Model ===")
    corpus = [
        "the cat sat on the mat",
        "the dog sat on the rug",
        "the cat ate the rat",
        "the rat ran on the mat",
        "a cat is a mammal",
    ]
    # Build bigram model
    from collections import defaultdict, Counter
    bigrams = defaultdict(Counter)
    for sent in corpus:
        words = ["<s>"] + sent.split() + ["</s>"]
        for i in range(len(words)-1):
            bigrams[words[i]][words[i+1]] += 1

    # Convert to probabilities
    vocab = sorted(set(w for s in corpus for w in s.split()) | {"<s>", "</s>"})
    V = len(vocab)

    def bigram_prob(w1, w2, alpha=0.1):   # Laplace smoothing
        count = bigrams[w1][w2] + alpha
        total = sum(bigrams[w1].values()) + alpha * V
        return count / total

    # Perplexity on a test sentence
    test = "the cat sat on the rug"
    words = ["<s>"] + test.split() + ["</s>"]
    log_p = sum(np.log(bigram_prob(words[i], words[i+1])) for i in range(len(words)-1))
    ppl   = np.exp(-log_p / (len(words)-1))
    print(f"  Bigram model | Vocab: {V}  Corpus: {len(corpus)} sentences")
    print(f"  Test: '{test}'")
    print(f"  Log-prob: {log_p:.4f}  Perplexity: {ppl:.2f}")

    # Generate a sentence
    def generate(n=10, start="<s>"):
        rng = np.random.default_rng(42)
        w = start; sent = []
        for _ in range(n):
            opts = [(w2, bigram_prob(w, w2)) for w2 in vocab]
            words2, probs = zip(*opts)
            probs = np.array(probs); probs /= probs.sum()
            w = rng.choice(words2, p=probs)
            if w == "</s>": break
            sent.append(w)
        return " ".join(sent)

    print(f"  Generated: '{generate()}'")


# -- 3. Transformer-based autoregressive LM ------------------------------------
def transformer_lm_overview():
    print("\n=== Transformer Autoregressive LM (GPT-style) ===")
    print()
    print("  Architecture:")
    print("    Input tokens -> embedding + positional -> N x decoder blocks -> logits")
    print()
    print("  Decoder block:")
    print("    1. Masked multi-head self-attention (causal mask)")
    print("    2. Add & LayerNorm")
    print("    3. Feed-forward (2-layer MLP, GELU)")
    print("    4. Add & LayerNorm")
    print()
    print("  Causal mask: attention only to left context")
    print("    A[i,j] = -inf if j > i  (prevents peeking at future)")
    print()

    # Simulate GPT-2 scale
    configs = [
        ("GPT-2 small",   117, 12, 12, 768,  "OpenAI 2019"),
        ("GPT-2 large",   774, 36, 20, 1280, "OpenAI 2019"),
        ("GPT-3 6.7B",   6700, 32, 32, 4096, "OpenAI 2020"),
        ("LLaMA-7B",     7000, 32, 32, 4096, "Meta 2023"),
        ("GPT-4",      >1000,  "?", "?", "?",  "OpenAI 2023 (MoE rumoured)"),
        ("LLaMA-3.1-70B",70000, 80, 64, 8192, "Meta 2024"),
    ]
    print(f"  {'Model':<18} {'Params(M)':>10} {'Layers':>7} {'Heads':>6} {'d_model':>8}")
    for row in configs:
        m, p, l, h, d, note = row
        print(f"  {m:<18} {str(p):>10} {str(l):>7} {str(h):>6} {str(d):>8}")


# -- 4. Decoding strategies ----------------------------------------------------
def decoding_strategies():
    print("\n=== Decoding Strategies ===")
    V = 20  # vocabulary size
    rng = np.random.default_rng(0)
    logits = rng.standard_normal(V)
    logits[3] += 3; logits[7] += 2; logits[12] += 1.5  # peak at 3, 7, 12
    probs = softmax(logits)

    print(f"  Vocabulary size: {V}")
    print(f"  Top-5 tokens: {np.argsort(probs)[::-1][:5]}")
    print(f"  Top-5 probs:  {np.sort(probs)[::-1][:5].round(4)}")
    print()

    # 1. Greedy
    greedy = probs.argmax()
    print(f"  Greedy:          token {greedy}  (always picks argmax)")

    # 2. Temperature sampling
    for T in [0.5, 1.0, 2.0]:
        probs_t = softmax(logits / T)
        samples = [np.random.default_rng(i).choice(V, p=probs_t) for i in range(5)]
        print(f"  Temp={T:.1f}:         samples={samples}  entropy={-(probs_t*np.log(probs_t+1e-9)).sum():.3f}")

    # 3. Top-k sampling
    k = 5
    top_k_idx = np.argsort(probs)[::-1][:k]
    probs_k   = probs[top_k_idx]; probs_k /= probs_k.sum()
    sample_k  = np.random.default_rng(0).choice(top_k_idx, p=probs_k)
    print(f"  Top-k (k={k}):     token {sample_k}  (from {top_k_idx})")

    # 4. Top-p (nucleus)
    p_thresh = 0.9
    sorted_idx   = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumprobs     = sorted_probs.cumsum()
    nucleus      = sorted_idx[cumprobs <= p_thresh]
    if len(nucleus) == 0: nucleus = sorted_idx[:1]
    probs_p = probs[nucleus]; probs_p /= probs_p.sum()
    sample_p = np.random.default_rng(0).choice(nucleus, p=probs_p)
    print(f"  Top-p (p={p_thresh}):   token {sample_p}  (nucleus size: {len(nucleus)})")

    # 5. Beam search explanation
    print()
    print("  Beam search (beam_size B):")
    print("    Maintain top-B partial sequences at each step")
    print("    Select B×V candidates, keep best B")
    print("    Returns best complete sequence by log-prob")
    print("    B=1: greedy  B=inf: exact search (intractable)")


# -- 5. PixelCNN ---------------------------------------------------------------
def pixelcnn_overview():
    print("\n=== PixelCNN ===")
    print("  van den Oord et al. (2016)")
    print("  Autoregressive image generation: pixel by pixel, raster order")
    print()
    print("  Masked convolution:")
    print("    Kernel mask ensures pixel (i,j) only sees pixels above and to the left")
    print("    Type A mask: excludes current pixel (first layer)")
    print("    Type B mask: includes current pixel (subsequent layers)")
    print()

    # Visualise 5×5 mask
    kH = kW = 5; centre = kH // 2
    mask_A = np.zeros((kH, kW), dtype=int)
    for i in range(kH):
        for j in range(kW):
            if i < centre or (i == centre and j < centre):
                mask_A[i, j] = 1
    mask_B = mask_A.copy(); mask_B[centre, centre] = 1

    print("  Type A mask (5×5):")
    for row in mask_A:
        print(f"    {' '.join(['■' if v else '□' for v in row])}")
    print()
    print("  Type B mask (5×5):")
    for row in mask_B:
        print(f"    {' '.join(['■' if v else '□' for v in row])}")
    print()
    print("  PixelCNN++ improvements:")
    print("    Discretised logistic mixture -> better density")
    print("    Skip connections (ResNet style)")
    print("    Downsampling paths for larger receptive field")


# -- 6. WaveNet overview -------------------------------------------------------
def wavenet_overview():
    print("\n=== WaveNet (Audio Autoregressive) ===")
    print("  van den Oord et al. (2016) — raw audio waveform generation")
    print()
    print("  Key ideas:")
    print("    Dilated causal convolutions — exponentially growing receptive field")
    print("    Stack L dilated conv layers: d = 1, 2, 4, 8, ..., 2^(L-1)")
    print("    Receptive field = (2^L - 1) × N blocks + 1")
    print()
    print("  Dilation example:")
    for d in [1, 2, 4, 8, 16]:
        receptive = d * 2 + 1
        print(f"    dilation={d:>2}: receptive field = {receptive} samples  "
              f"({receptive/16000*1000:.2f}ms at 16kHz)")
    print()
    print("  With 3 stacks of L=10 layers: receptive field = 3×1023+1 = 3070 samples")
    print("  (~190ms at 16kHz) — sufficient for speech")
    print()
    print("  µ-law quantisation: 256 categories (8-bit audio)")
    print("  Generation: slow (16000 steps per second of audio)")
    print("  Parallel WaveNet / WaveGlow: much faster inference via distillation")


if __name__ == "__main__":
    autoregressive_principle()
    ngram_lm_demo()
    transformer_lm_overview()
    decoding_strategies()
    pixelcnn_overview()
    wavenet_overview()
