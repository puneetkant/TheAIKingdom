"""
Working Example: RL Environments and Tools
Covers Gymnasium, common environments, reward shaping,
evaluation, and major RL libraries.
"""
import numpy as np
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_rl_tools")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Gymnasium overview ─────────────────────────────────────────────────────
def gymnasium_overview():
    print("=== Gymnasium (OpenAI Gym) ===")
    print("  Standard interface for RL environments")
    print("  pip install gymnasium")
    print()
    print("  Core API:")
    code = [
        ("import gymnasium as gym",                   "Import"),
        ("env = gym.make('CartPole-v1')",             "Create env"),
        ("obs, info = env.reset(seed=42)",            "Reset; get initial obs"),
        ("action = env.action_space.sample()",        "Random action"),
        ("obs, reward, terminated, truncated, info = env.step(action)",
                                                      "Take step"),
        ("done = terminated or truncated",            "Episode ended"),
        ("env.close()",                               "Clean up"),
    ]
    for c, d in code:
        print(f"  {c:<60} # {d}")
    print()
    print("  Space types:")
    spaces = [
        ("Discrete(n)",             "n discrete actions (0..n-1)"),
        ("Box(low, high, shape)",   "Continuous; real-valued tensors"),
        ("MultiBinary(n)",          "n binary switches"),
        ("MultiDiscrete([n1,n2])",  "Multiple discrete axes"),
        ("Dict({...})",             "Dictionary of sub-spaces"),
        ("Tuple((...,))",           "Tuple of sub-spaces"),
    ]
    for s, d in spaces:
        print(f"    {s:<30} {d}")

    print()
    try:
        import gymnasium as gym
        env  = gym.make("CartPole-v1")
        obs, info = env.reset(seed=0)
        print(f"  CartPole-v1 live demo:")
        print(f"    obs_space:    {env.observation_space}")
        print(f"    act_space:    {env.action_space}")
        print(f"    Initial obs:  {obs.round(4)}")
        total_r = 0
        for _ in range(200):
            a = env.action_space.sample()
            obs, r, term, trunc, _ = env.step(a)
            total_r += r
            if term or trunc: break
        env.close()
        print(f"    Random policy episode return: {total_r:.1f}")
    except ImportError:
        print("  gymnasium not installed; showing code pattern only")


# ── 2. Classic control environments ──────────────────────────────────────────
def classic_environments():
    print("\n=== Common RL Environments ===")
    print()
    envs = [
        ("Classic Control", [
            ("CartPole-v1",         "Balance pole; 4-D obs; 2 actions; max 500 steps"),
            ("MountainCar-v0",      "Push car uphill; sparse reward; hard exploration"),
            ("Pendulum-v1",         "Continuous; swing up + hold upright"),
            ("Acrobot-v1",          "2-link pendulum; swing tip above target"),
        ]),
        ("Box2D", [
            ("LunarLander-v2",      "Land on pad; discrete or continuous"),
            ("BipedalWalker-v3",    "Continuous; walk on rough terrain"),
            ("CarRacing-v2",        "96×96 pixel input; continuous steering"),
        ]),
        ("Atari (ALE)", [
            ("ALE/Breakout-v5",     "Pixel input; played by DQN 2015"),
            ("ALE/Pong-v5",         "Table tennis; solved by simple PG"),
            ("ALE/Montezuma",       "Hard exploration; 8 rooms; sparse reward"),
        ]),
        ("MuJoCo", [
            ("HalfCheetah-v4",      "Fast running; continuous; SAC/PPO"),
            ("Ant-v4",              "4-legged locomotion; 111-D obs"),
            ("Humanoid-v4",         "High-dim; 376-D obs; hardest locomotion"),
        ]),
        ("Multi-Agent", [
            ("PettingZoo",          "Multi-agent games, adversarial, cooperative"),
            ("StarCraft2 (SMAC)",   "Micromanagement; Dec-POMDP"),
        ]),
    ]
    for category, items in envs:
        print(f"  {category}:")
        for name, desc in items:
            print(f"    {name:<28} {desc}")
        print()


# ── 3. Reward shaping ────────────────────────────────────────────────────────
def reward_shaping():
    print("=== Reward Shaping ===")
    print("  Sparse rewards are hard to learn from → add additional reward signals")
    print()
    print("  Potential-based shaping (Ng et al.):")
    print("    R_shaped = R + γΦ(s') - Φ(s)")
    print("    Guaranteed to preserve optimal policy when Φ is a potential fn")
    print()

    techniques = [
        ("Potential shaping",    "Distance to goal, normalised height, etc."),
        ("Reward clipping",      "Clip to [-1,1] for numerical stability (DQN)"),
        ("Normalisation",        "Running mean/std normalisation of rewards"),
        ("Intrinsic curiosity",  "Bonus for visiting novel states (ICM)"),
        ("Count-based",          "Bonus ∝ 1/√N(s); tabular exploration"),
        ("RND",                  "Random Network Distillation; neural novelty bonus"),
        ("HER",                  "Hindsight Experience Replay; relabel failed goals"),
        ("RLHF",                 "Reward from human preference comparisons"),
    ]
    for t, d in techniques:
        print(f"  {t:<22} {d}")

    # Simulate effect of reward shaping on MountainCar-like
    print()
    print("  Toy example: sparse vs shaped reward")
    n_episodes = 50; rng = np.random.default_rng(0)
    sparse  = [rng.integers(1, 5) if ep > 40 else 0 for ep in range(n_episodes)]
    shaped  = [rng.integers(1, 5) + rng.uniform(0, 3) for ep in range(n_episodes)]
    print(f"  Sparse  avg (ep 1-10): {np.mean(sparse[:10]):.2f}  (ep 40-50): {np.mean(sparse[40:]):.2f}")
    print(f"  Shaped  avg (ep 1-10): {np.mean(shaped[:10]):.2f}  (ep 40-50): {np.mean(shaped[40:]):.2f}")


# ── 4. RL evaluation best practices ──────────────────────────────────────────
def evaluation_best_practices():
    print("\n=== RL Evaluation Best Practices ===")
    print("  Common pitfalls and how to avoid them:")
    print()
    practices = [
        ("Use multiple seeds",       "Report mean ± std over ≥5 seeds; single seed misleading"),
        ("Evaluation episodes",      "Evaluate with no exploration (ε=0 or deterministic)"),
        ("Report learning curves",   "Plot return vs environment steps, not wall time"),
        ("Confidence intervals",     "95% CI or interquartile mean (IQM) across seeds"),
        ("Normalised scores",        "Score = (agent - random) / (expert - random)"),
        ("Train/test gap",           "Evaluate on held-out seeds/scenarios"),
        ("Compute budget",           "Report total environment steps, not just episodes"),
        ("Compare at same steps",    "Compare algorithms at equal environment interaction"),
    ]
    for p, d in practices:
        print(f"  {p:<22} {d}")


# ── 5. RL libraries ───────────────────────────────────────────────────────────
def rl_libraries():
    print("\n=== RL Libraries ===")
    libs = [
        ("Stable-Baselines3",  "sb3",  "PPO, A2C, SAC, TD3, DDPG; easy API; PyTorch"),
        ("RLlib",              "ray[rllib]",  "Ray-based; scalable; multi-agent; many algos"),
        ("CleanRL",            "cleanrl","Single-file implementations; educational; reproducible"),
        ("Tianshou",           "tianshou","Modular; Torch; 20+ algos; good docs"),
        ("Sample Factory",     "sample-factory","Extreme throughput; APPO; Atari/vizdoom"),
        ("Acme (DeepMind)",    "dm-acme","Distributed RL; JAX/TF; research-grade"),
        ("OpenRL",             "openrl","Multi-agent; PettingZoo; recent"),
        ("TorchRL",            "torchrl","PyTorch official; modular; composable"),
    ]
    print(f"  {'Library':<18} {'Install':<22} {'Notes'}")
    print(f"  {'─'*18} {'─'*22} {'─'*40}")
    for name, install, desc in libs:
        print(f"  {name:<18} pip install {install:<12} {desc}")

    print()
    print("  Stable-Baselines3 minimal example:")
    code = """
    from stable_baselines3 import PPO
    import gymnasium as gym

    env   = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)

    obs, _ = env.reset()
    for _ in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)
        if term or trunc: obs, _ = env.reset()
    """
    for line in code.strip().split("\n"):
        print(f"    {line}")

    try:
        from stable_baselines3 import PPO
        import gymnasium as gym
        env   = gym.make("CartPole-v1")
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=5000)
        obs, _ = env.reset()
        rewards = []
        for ep in range(5):
            obs, _ = env.reset(); ep_r = 0
            for _ in range(500):
                a, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, _ = env.step(a)
                ep_r += r
                if term or trunc: break
            rewards.append(ep_r)
        env.close()
        print(f"\n  SB3 PPO (5000 steps) eval over 5 eps: {np.round(rewards, 1)}")
    except ImportError:
        print("\n  stable-baselines3 not installed (pip install stable-baselines3)")


if __name__ == "__main__":
    gymnasium_overview()
    classic_environments()
    reward_shaping()
    evaluation_best_practices()
    rl_libraries()
