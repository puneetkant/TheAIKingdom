"""
Working Example: Supervised Fine-Tuning (SFT)
Covers instruction tuning, dataset preparation, chat templates,
catastrophic forgetting, and evaluation.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_sft")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. What is SFT? -----------------------------------------------------------
def sft_overview():
    print("=== Supervised Fine-Tuning (SFT) ===")
    print()
    print("  SFT: continue training a pre-trained LLM on")
    print("  curated (instruction, response) pairs")
    print()
    print("  Pre-training -> SFT -> RLHF -> Deployment")
    print()
    print("  Why SFT?")
    why = [
        ("Instruction following","Teach model to respond to user intent"),
        ("Format adherence",     "JSON output, markdown, code blocks"),
        ("Safety",               "Refuse harmful requests; be helpful"),
        ("Domain adaptation",    "Medical, legal, coding specialisation"),
        ("Chat interface",       "Multi-turn conversation template"),
    ]
    for w, d in why:
        print(f"  {w:<22} {d}")

    print()
    print("  Key datasets:")
    datasets = [
        ("Alpaca",          "52k GPT-3.5 generated instructions; cheap but noisy"),
        ("ShareGPT",        "Human ChatGPT conversations; high quality"),
        ("FLAN collection", "1.8k tasks; chain-of-thought; multi-task"),
        ("OpenHermes 2.5",  "1M synthetic; GPT-4 generated; strong instruct"),
        ("Magpie",          "Self-synthesis; no human labels needed"),
        ("UltraChat",       "Large multi-turn; diverse domains"),
        ("OpenAssistant",   "Human-annotated; OASST2; multi-lingual"),
    ]
    for d, desc in datasets:
        print(f"  {d:<18} {desc}")


# -- 2. Chat templates ---------------------------------------------------------
def chat_templates():
    print("\n=== Chat Templates ===")
    print()
    print("  Chat templates structure multi-turn conversations")
    print("  Different models use different formats — must match training format")
    print()
    print("  Llama-3 format (tiktoken):")
    llama3 = '''
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The capital of France is Paris.<|eot_id|>
'''
    print(llama3)
    print("  ChatML format (OpenAI, Mistral):")
    chatml = '''
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
'''
    print(chatml)
    print("  Key: only compute loss on assistant tokens (not user/system)")


# -- 3. Loss masking -----------------------------------------------------------
def loss_masking():
    print("\n=== SFT Loss Masking ===")
    print()
    print("  Full sequence loss (naive):  compute loss on all tokens")
    print("  Response-only loss (proper): mask prompt tokens; only backprop on response")
    print()
    print("  Why it matters:")
    print("    Training on prompt tokens wastes compute and can hurt performance")
    print("    Model should learn to generate, not to memorise prompts")
    print()

    # Simulate tokenised conversation
    tokens =      ["<sys>", "help", "</sys>", "<usr>", "capital", "France", "</usr>", "<ast>", "Paris", "</ast>"]
    is_assistant = [False,  False,  False,    False,   False,     False,    False,    False,   True,    True]
    labels =       [-100,   -100,   -100,     -100,    -100,      -100,     -100,     -100,    1543,    151]

    print(f"  {'Token':<12} {'Is assistant':<14} {'Label (CE loss)'}")
    for tok, is_ast, lbl in zip(tokens, is_assistant, labels):
        lbl_str = str(lbl) if lbl != -100 else "-100 (masked)"
        print(f"  {tok:<12} {str(is_ast):<14} {lbl_str}")
    print()
    print("  -100 labels are ignored by PyTorch's CrossEntropyLoss")


# -- 4. Catastrophic forgetting ------------------------------------------------
def catastrophic_forgetting():
    print("\n=== Catastrophic Forgetting ===")
    print()
    print("  Fine-tuning on narrow data -> model forgets base capabilities")
    print()
    print("  Mitigation strategies:")
    strategies = [
        ("Low learning rate",    "1e-5 to 5e-6; prevents large weight updates"),
        ("Few epochs",           "1-3 epochs; stop before overfitting"),
        ("Data mixing",          "Mix SFT data with pre-training data (5-20%)"),
        ("PEFT (LoRA)",          "Only train small adapter; base weights frozen"),
        ("EWC",                  "Elastic Weight Consolidation; penalise drift from important weights"),
        ("Replay buffer",        "Store pre-training examples; interleave"),
        ("Eval on base tasks",   "Monitor MMLU/HumanEval; stop if degrading"),
    ]
    for s, d in strategies:
        print(f"  {s:<24} {d}")

    print()
    print("  SFT hyperparameter recommendations:")
    hparams = [
        ("Learning rate",   "1e-5 to 2e-5 (full FT); 2e-4 (LoRA)"),
        ("LR schedule",     "Cosine with warmup (3-5% steps)"),
        ("Batch size",      "64-256 (effective); gradient accumulation"),
        ("Epochs",          "1-3; early stopping on val loss"),
        ("Max seq length",  "Match use case; 2k-8k common"),
        ("Padding",         "Pack multiple examples per sequence (no waste)"),
    ]
    for h, d in hparams:
        print(f"  {h:<16} {d}")


if __name__ == "__main__":
    sft_overview()
    chat_templates()
    loss_masking()
    catastrophic_forgetting()
