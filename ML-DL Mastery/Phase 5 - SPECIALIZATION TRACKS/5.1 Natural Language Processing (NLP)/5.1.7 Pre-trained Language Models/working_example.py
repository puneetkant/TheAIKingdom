"""
Working Example: Pre-trained Language Models
Covers BERT fine-tuning concepts, GPT-style generation, transfer learning
for NLP, and common fine-tuning strategies.
Runs with sklearn only; shows HuggingFace code patterns as printed examples.
"""
import numpy as np
import re, math
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# ── 1. Pre-training vs Fine-tuning ────────────────────────────────────────────
def pretraining_vs_finetuning():
    print("=== Pre-training vs Fine-tuning ===")
    print()
    print("  Pre-training (self-supervised, no labels):")
    print("    BERT: Masked Language Modelling (MLM) + Next Sentence Prediction (NSP)")
    print("          → 'The [MASK] sat on the mat'  → predict 'cat'")
    print("    GPT:  Autoregressive LM (causal)")
    print("          → 'The cat sat on the'  → predict 'mat'")
    print("    T5:   Text-to-text (all tasks as seq2seq)")
    print()
    print("  Fine-tuning (supervised, with task labels):")
    print("    Add a task-specific head on top of the pre-trained model")
    print("    Fine-tune ALL parameters (or subset with PEFT)")
    print()
    print("  Fine-tuning stages:")
    print("    1. Full fine-tuning: update all layers (expensive)")
    print("    2. Head-only:        freeze backbone, train only classifier head")
    print("    3. Discriminative:   lower LR for earlier layers")
    print("    4. PEFT (LoRA, etc.): train only small adapters (<1% params)")


# ── 2. BERT fine-tuning patterns ─────────────────────────────────────────────
def bert_finetuning():
    print("\n=== BERT Fine-tuning Code Patterns ===")
    print("""
  from transformers import BertTokenizer, BertForSequenceClassification
  import torch

  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  model     = BertForSequenceClassification.from_pretrained(
                  'bert-base-uncased', num_labels=2)

  # Tokenise
  enc = tokenizer(texts, truncation=True, padding=True,
                  max_length=128, return_tensors='pt')
  # enc: {input_ids, attention_mask, token_type_ids}

  # Fine-tune
  optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
  for epoch in range(3):
      outputs = model(**enc, labels=labels)
      loss    = outputs.loss
      loss.backward()
      optimizer.step(); optimizer.zero_grad()

  # Inference
  model.eval()
  with torch.no_grad():
      logits = model(**enc).logits
      preds  = logits.argmax(-1)
    """)

    print("  BERT input format:")
    print("    [CLS] sentence A [SEP] sentence B [SEP]")
    print("    The [CLS] token representation is used for classification")
    print()
    print("  Common BERT variants:")
    rows = [
        ("BERT-base",      "110M params", "12L, 768H, 12A"),
        ("BERT-large",     "340M params", "24L, 1024H, 16A"),
        ("RoBERTa-base",   "125M params", "No NSP, more data, longer training"),
        ("DistilBERT",     "66M params",  "60% smaller, 97% of BERT perf"),
        ("ALBERT",         "12M params",  "Weight sharing + sentence-order prediction"),
        ("DeBERTa-v3-base","86M params",  "Disentangled attention, best encoder"),
    ]
    print(f"  {'Model':<18} {'Size':<15} Details")
    print(f"  {'─'*18} {'─'*15} {'─'*35}")
    for m, s, d in rows:
        print(f"  {m:<18} {s:<15} {d}")


# ── 3. GPT-style generation ───────────────────────────────────────────────────
def gpt_generation_patterns():
    print("\n=== GPT-style Text Generation ===")
    print("""
  from transformers import GPT2Tokenizer, GPT2LMHeadModel

  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  model     = GPT2LMHeadModel.from_pretrained('gpt2')

  prompt = "The future of artificial intelligence is"
  inputs = tokenizer(prompt, return_tensors='pt')

  # Greedy decoding
  out = model.generate(**inputs, max_new_tokens=50)

  # Beam search
  out = model.generate(**inputs, num_beams=5, early_stopping=True)

  # Sampling (temperature + top-p)
  out = model.generate(**inputs, do_sample=True, temperature=0.8,
                        top_p=0.9, top_k=50, max_new_tokens=100)

  print(tokenizer.decode(out[0], skip_special_tokens=True))
    """)

    print("  Decoding strategies:")
    strategies = [
        ("Greedy",       "argmax at each step; fast but repetitive"),
        ("Beam search",  "keep top-k sequences; good for translation"),
        ("Temperature",  "T<1 → sharper; T>1 → more random"),
        ("Top-k",        "sample from top-k tokens (k=50)"),
        ("Top-p (nucleus)", "sample from smallest set with Σ P ≥ p"),
        ("Rep. penalty", "penalise already-generated tokens"),
    ]
    for s, d in strategies:
        print(f"    {s:<20} {d}")


# ── 4. Simulated feature extraction (BERT-like) ───────────────────────────────
def simulated_bert_features():
    """
    Simulates using pre-trained features for downstream classification.
    In reality these would be BERT embeddings; here we use TF-IDF as proxy.
    """
    print("\n=== Simulated PLM Feature Extraction Demo ===")
    from sklearn.feature_extraction.text import TfidfVectorizer

    pos_texts = [
        "This movie is absolutely fantastic and entertaining",
        "A wonderful film with excellent performances",
        "Loved the story and the amazing cinematography",
        "Brilliant acting and superb direction throughout",
        "One of the best films I have seen this year",
        "Thoroughly enjoyed this beautifully crafted movie",
    ]
    neg_texts = [
        "Terrible film complete waste of two hours",
        "Boring predictable and poorly written script",
        "The worst movie I have seen in years awful",
        "Dreadful acting and a painfully slow plot",
        "Completely disappointed with this awful production",
        "Hated every scene of this horrible film",
    ]

    texts  = pos_texts + neg_texts
    labels = [1]*len(pos_texts) + [0]*len(neg_texts)

    # Simulate "BERT-like" features with TF-IDF (proxy)
    vec = TfidfVectorizer(ngram_range=(1, 2))
    X   = vec.fit_transform(texts).toarray()

    # Head-only (frozen backbone simulation)
    np.random.seed(0)
    idx = np.random.permutation(len(texts))
    Xtr, ytr = X[idx[:8]], [labels[i] for i in idx[:8]]
    Xts, yts = X[idx[8:]], [labels[i] for i in idx[8:]]

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=0)
    clf.fit(Xtr, ytr)
    yp  = clf.predict(Xts)
    acc = (yp == np.array(yts)).mean()
    print(f"  Proxy TF-IDF + LogReg (head-only):")
    print(f"  Features: {X.shape[1]}  Train: {len(Xtr)}  Test: {len(Xts)}")
    print(f"  Accuracy: {acc:.4f}")
    print()
    print("  With real BERT embeddings (768-dim):")
    print("    → typically 90-95% accuracy on SST-2 with only 10 examples (few-shot)")


# ── 5. Transfer learning strategies ──────────────────────────────────────────
def transfer_learning_nlp():
    print("\n=== Transfer Learning Strategies for NLP ===")
    strategies = [
        ("Zero-shot",         "No fine-tuning; prompt the model directly",
         "GPT-4, Claude — 'Is this review positive or negative?'"),
        ("Few-shot",          "Provide K examples in the prompt",
         "GPT-3 in-context learning with 5-20 examples"),
        ("Feature extract",   "Freeze model; train classifier on embeddings",
         "BERT-base + LogReg; fast, low compute"),
        ("Full fine-tuning",  "Update all parameters",
         "BERT-base on GLUE tasks; best accuracy, expensive"),
        ("LoRA / QLoRA",      "Low-rank adapters (0.1-1% of params)",
         "Fine-tune 7B LLaMA on single GPU"),
        ("Prompt tuning",     "Learn soft prompt tokens (frozen LLM)",
         "T5 prompt tuning; parameter-free adaptation"),
        ("Instruction tuning","Fine-tune on (instruction, output) pairs",
         "FLAN-T5, Alpaca — generalist task following"),
    ]
    print(f"  {'Strategy':<18} {'Description':<38} Example")
    print(f"  {'─'*18} {'─'*38} {'─'*35}")
    for s, d, e in strategies:
        print(f"  {s:<18} {d:<38} {e[:35]}")

    print()
    print("  Rule of thumb:")
    print("    <100 examples:    zero/few-shot or head-only fine-tuning")
    print("    100-10K examples: full fine-tuning or LoRA")
    print("    >10K examples:    full fine-tuning, possibly train from scratch")


if __name__ == "__main__":
    pretraining_vs_finetuning()
    bert_finetuning()
    gpt_generation_patterns()
    simulated_bert_features()
    transfer_learning_nlp()
