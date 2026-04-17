"""
Working Example 2: Transformer Variants — BERT vs GPT vs T5 architecture comparison
======================================================================================
Demonstrates the encoder-only, decoder-only, encoder-decoder paradigms.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
except ImportError:
    raise SystemExit("pip install numpy")

def demo():
    print("=== Transformer Variants Comparison ===\n")
    variants = [
        {
            "name": "BERT (Encoder-only)",
            "masking": "Bidirectional (all positions see all)",
            "pretraining": "Masked Language Modeling + NSP",
            "use_case": "Classification, NER, QA (understanding)",
            "key_models": "BERT, RoBERTa, DistilBERT, ALBERT, DeBERTa",
        },
        {
            "name": "GPT (Decoder-only)",
            "masking": "Causal (each token sees only past)",
            "pretraining": "Next token prediction (CLM)",
            "use_case": "Text generation, chat, code (generation)",
            "key_models": "GPT-2, GPT-3/4, LLaMA, Mistral, Falcon",
        },
        {
            "name": "T5 / BART (Encoder-Decoder)",
            "masking": "Encoder: bidirectional; Decoder: causal",
            "pretraining": "Span masking / Denoising",
            "use_case": "Translation, summarization, seq2seq",
            "key_models": "T5, BART, Pegasus, mT5",
        },
    ]
    for v in variants:
        print(f"  {'='*55}")
        for k, val in v.items():
            print(f"  {k:15s}: {val}")
    print()

    # Causal mask vs no mask (quick numpy demo)
    print("=== Attention mask demo ===")
    n = 5
    # Causal (GPT)
    causal_mask = np.tril(np.ones((n, n)))
    # Bidirectional (BERT)
    bidi_mask = np.ones((n, n))
    print("  Causal mask (GPT/decoder):\n", causal_mask.astype(int))
    print("  Bidirectional (BERT/encoder):\n", bidi_mask.astype(int))

if __name__ == "__main__":
    demo()
