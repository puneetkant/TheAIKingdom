"""
Working Example: Embodied AI and Robotics
Covers robot learning, foundation models for robotics, sim-to-real transfer,
and manipulation/navigation tasks.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_robotics")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Embodied AI overview ───────────────────────────────────────────────────
def embodied_ai_overview():
    print("=== Embodied AI and Robotics ===")
    print()
    print("  Embodied AI: AI agents that perceive and act in physical or simulated environments")
    print("  Differs from pure LLMs: continuous action spaces, partial observability, real-time")
    print()
    print("  Core tasks:")
    tasks = [
        ("Robot manipulation", "Pick-and-place, assembly, tool use"),
        ("Navigation",         "Navigate rooms, outdoor, social spaces"),
        ("Loco-manipulation",  "Walk + manipulate (quadruped + arm)"),
        ("Sim-to-real",        "Train in simulation; deploy on real robot"),
        ("Language grounding", "Follow natural language instructions"),
        ("Human-robot collab", "Work alongside humans safely"),
    ]
    for t, d in tasks:
        print(f"  {t:<24} {d}")


# ── 2. Foundation models for robotics ─────────────────────────────────────────
def robot_foundation_models():
    print("\n=== Foundation Models for Robotics ===")
    print()
    models = [
        ("RT-1",     "Google; 130k real demos; 97% success; transformer action model"),
        ("RT-2",     "Google; VLM (PaLM-E); VQA → robot actions; web knowledge"),
        ("OpenVLA",  "Open; LLaMA-2 + SigLIP; visual language action model"),
        ("π0",       "Physical Intelligence; flow matching; dexterous manipulation"),
        ("π0.5",     "Physical Intelligence; 7B VLA + low-level expert policies"),
        ("GROOT",    "NVIDIA; video prediction → generalised manipulation"),
        ("GR-1",     "Video-to-action; predict future frames; goal-conditioned"),
        ("UniSim",   "Real-world simulator; learn from video; no robot needed"),
        ("SayCan",   "Google; LLM affordance-grounded task planning"),
        ("PaLM-E",   "562B embodied multimodal model; robot + NLU + vision"),
    ]
    print(f"  {'Model':<12} {'Notes'}")
    for m, d in models:
        print(f"  {m:<12} {d}")


# ── 3. Sim-to-real transfer ───────────────────────────────────────────────────
def sim_to_real():
    print("\n=== Sim-to-Real Transfer ===")
    print()
    print("  Simulation gap: simulator physics ≠ real world")
    print()
    techniques = [
        ("Domain randomisation","Vary simulator parameters at train time; robust policy"),
        ("Domain adaptation",   "Match sim→real distribution; pixel-level adaptation"),
        ("Photo-realistic sim", "USD/NVIDIA Omniverse; physically accurate rendering"),
        ("Privileged learning", "Teacher (access sim state) → Student (only observations)"),
        ("Real2Sim2Real",       "Scan real object → sim asset → train → deploy"),
        ("Residual learning",   "Learn delta from sim policy to compensate real offset"),
    ]
    print(f"  {'Technique':<24} {'Description'}")
    for t, d in techniques:
        print(f"  {t:<24} {d}")

    print()
    print("  Simulators:")
    sims = [
        ("Isaac Gym / Lab",    "NVIDIA; GPU parallelism; 1000s envs simultaneously"),
        ("MuJoCo",             "DeepMind; gold standard for locomotion research"),
        ("PyBullet",           "Open-source; easy to use; slower"),
        ("Sapien",             "Articulated objects; manipulation focus; open"),
        ("Genesis",            "New; differentiable; 430k sim steps/sec on single GPU"),
    ]
    for s, d in sims:
        print(f"  {s:<18} {d}")


# ── 4. Imitation and RL for robots ────────────────────────────────────────────
def robot_learning():
    print("\n=== Robot Learning Algorithms ===")
    print()
    print("  Imitation Learning (IL):")
    il_methods = [
        ("BC",         "Behavioural Cloning; supervised on demos; distribution shift"),
        ("DAgger",     "Iterative; query expert on visited states; safe BC"),
        ("GAIL",       "GAN-style; match state-action distribution"),
        ("ACT",        "Action Chunking with Transformers; 10-100 Hz; dexterous"),
        ("Diffusion Policy","Diffuse action trajectories; multimodal; robust"),
    ]
    for m, d in il_methods:
        print(f"  {m:<18} {d}")

    print()
    print("  Reinforcement Learning for robots:")
    rl_methods = [
        ("SAC",        "Off-policy; entropy; dexterous hand tasks"),
        ("PPO",        "On-policy; locomotion; stable but sample-hungry"),
        ("DDPG/TD3",   "Deterministic policy; continuous control"),
        ("Dreamer",    "World model; dream rollouts; efficient real-world RL"),
        ("Foundation+RL","Pre-train on demos (BC); fine-tune with RL (π0 flow)"),
    ]
    for m, d in rl_methods:
        print(f"  {m:<18} {d}")

    print()
    # Simulate a simple Diffusion Policy concept
    print("  Diffusion Policy concept:")
    print("    Action a_0 = start from noise a_T")
    print("    Denoise: a_{t-1} = ε_θ(a_t, obs, t)  (conditioned on observation)")
    print("    T denoising steps → smooth, multi-modal action distribution")
    print("    Advantages: handles ambiguous demonstrations naturally")


# ── 5. Benchmark environments ─────────────────────────────────────────────────
def robot_benchmarks():
    print("\n=== Robot Benchmarks ===")
    print()
    benchmarks = [
        ("Meta-World",    "50 manipulation tasks; MuJoCo; multi-task learning"),
        ("FurnitureBench", "IKEA furniture assembly; dexterous; contact-rich"),
        ("ManipulaTHOR",  "AI2-THOR; pick-and-place navigation"),
        ("RoboSuite",     "IIWA/Panda; modular; benchmarked BC algorithms"),
        ("Open X-Embodiment","Google RT; 22 robots; 1M+ demos; cross-robot eval"),
        ("LIBERO",        "Language-conditioned; 130 tasks; lifelong learning"),
        ("BiGym",         "Bimanual manipulation; 40 tasks; humanoid"),
    ]
    print(f"  {'Benchmark':<18} {'Description'}")
    for b, d in benchmarks:
        print(f"  {b:<18} {d}")


if __name__ == "__main__":
    embodied_ai_overview()
    robot_foundation_models()
    sim_to_real()
    robot_learning()
    robot_benchmarks()
