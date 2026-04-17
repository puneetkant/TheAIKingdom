"""
Working Example: Key LLM Families
Covers GPT, LLaMA, Gemini, Claude, Mistral, and other major models.
"""
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_llm_families")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def llm_families():
    print("=== Key LLM Families ===")
    print()
    print("  ── OpenAI GPT Family ─────────────────────────────────────────────")
    gpt = [
        ("GPT-1",   "2018", "117M",   "Decoder-only; first large language model"),
        ("GPT-2",   "2019", "1.5B",   "Trained on WebText; zero-shot capabilities"),
        ("GPT-3",   "2020", "175B",   "Few-shot learning; InstructGPT fine-tuned"),
        ("GPT-3.5", "2022", "175B",   "RLHF; ChatGPT; widely deployed"),
        ("GPT-4",   "2023", "~1.8T?", "MoE; multimodal; MMLU 86.4%"),
        ("GPT-4o",  "2024", "~200B?", "Omni; native audio/vision; 200k context"),
        ("o1/o3",   "2024", "?",      "Reasoning chain; test-time compute scaling"),
    ]
    print(f"  {'Model':<10} {'Year'} {'Size':<10} {'Notes'}")
    for m, y, s, d in gpt:
        print(f"  {m:<10} {y}  {s:<10} {d}")

    print()
    print("  ── Meta LLaMA Family ─────────────────────────────────────────────")
    llama = [
        ("LLaMA-1",   "2023", "7–65B",  "Open weights; RoPE; RMSNorm; SwiGLU"),
        ("LLaMA-2",   "2023", "7–70B",  "GQA; RLHF chat; 4k context"),
        ("Code Llama","2023", "7–70B",  "Code-specialised; 100k context"),
        ("LLaMA-3",   "2024", "8–70B",  "128k context; 128k vocab; strong perf"),
        ("LLaMA-3.1", "2024", "8B–405B","128k context; multilingual; best open"),
        ("LLaMA-3.2", "2024", "1B–90B", "Multimodal vision; small edge models"),
    ]
    print(f"  {'Model':<12} {'Year'} {'Size':<12} {'Notes'}")
    for m, y, s, d in llama:
        print(f"  {m:<12} {y}  {s:<12} {d}")

    print()
    print("  ── Mistral / Mixtral Family ──────────────────────────────────────")
    mistral = [
        ("Mistral-7B",  "2023", "7B",    "Sliding window attention; fast; best 7B at launch"),
        ("Mixtral-8×7B","2023", "46.7B","Sparse MoE; 8 experts, top-2 active; 56 active params"),
        ("Mistral-Large","2024","123B",  "French; multilingual; strong reasoning"),
        ("Mistral-Nemo","2024", "12B",  "Tekken tokeniser; 128k context"),
    ]
    print(f"  {'Model':<16} {'Year'} {'Size':<8} {'Notes'}")
    for m, y, s, d in mistral:
        print(f"  {m:<16} {y}  {s:<8} {d}")

    print()
    print("  ── Google / DeepMind Family ──────────────────────────────────────")
    google = [
        ("PaLM",      "2022", "540B",  "Pathways; 780B tokens; few-shot SOTA"),
        ("PaLM-2",    "2023", "340B?", "Multilingual; efficient; powers Bard"),
        ("Gemini 1.0","2023", "?",     "Ultra/Pro/Nano; multimodal-first; native"),
        ("Gemini 1.5","2024", "?",     "1M+ context; MoE; efficient per token"),
        ("Gemini 2.0","2024", "?",     "Agentic; multimodal I/O; realtime"),
        ("Gemma",     "2024", "2–27B", "Open weights; MobileGemma for edge"),
    ]
    print(f"  {'Model':<14} {'Year'} {'Size':<8} {'Notes'}")
    for m, y, s, d in google:
        print(f"  {m:<14} {y}  {s:<8} {d}")

    print()
    print("  ── Anthropic Claude Family ───────────────────────────────────────")
    claude = [
        ("Claude-1",    "2023", "?",    "Constitutional AI; RLHF; helpful, harmless"),
        ("Claude-2",    "2023", "?",    "100k context; better coding and analysis"),
        ("Claude-3",    "2024", "?",    "Haiku/Sonnet/Opus; vision; 200k context"),
        ("Claude-3.5",  "2024", "?",    "Sonnet: best SWE-bench; Artifacts UI"),
        ("Claude-4",    "2025", "?",    "Extended thinking; frontier reasoning"),
    ]
    print(f"  {'Model':<14} {'Year'} {'Size':<6} {'Notes'}")
    for m, y, s, d in claude:
        print(f"  {m:<14} {y}  {s:<6} {d}")

    print()
    print("  ── Other Notable Models ──────────────────────────────────────────")
    others = [
        ("Falcon",        "TII UAE; 180B; open; 1T tokens"),
        ("Qwen-2.5",      "Alibaba; 0.5B–72B; multilingual; excellent coding"),
        ("DeepSeek-V3",   "DeepSeek; 671B MoE; 37B active; strong at maths/code"),
        ("DeepSeek-R1",   "Open reasoning model; o1-level; RL from scratch"),
        ("Phi-3/4",       "Microsoft; small but capable; 3.8B–14B; mobile-friendly"),
        ("Command R+",    "Cohere; enterprise; RAG-optimised; multilingual"),
        ("Yi-34B",        "01.AI; strong multilingual; 200k context"),
    ]
    for m, d in others:
        print(f"  {m:<16} {d}")


def capability_benchmarks():
    print("\n=== Key Benchmarks ===")
    print()
    benchmarks = [
        ("MMLU",        "57-subject multiple choice; knowledge breadth"),
        ("HumanEval",   "164 Python coding problems; pass@1"),
        ("MATH",        "12.5k competition math; exact answer"),
        ("GSM8K",       "8.5k grade school maths; chain of thought"),
        ("GPQA",        "Graduate-level science QA; hard"),
        ("SWE-bench",   "GitHub issue resolution; real codebases"),
        ("LMSYS Chatbot","ELO-rated human preference; head-to-head"),
        ("BIG-Bench Hard","23 hard tasks; emergent abilities"),
    ]
    print(f"  {'Benchmark':<18} {'Description'}")
    for b, d in benchmarks:
        print(f"  {b:<18} {d}")


if __name__ == "__main__":
    llm_families()
    capability_benchmarks()
