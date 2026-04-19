"""
Working Example: Reinforcement Learning from Human Feedback (RLHF)
Covers reward model training, PPO, and alignment pipeline.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_rlhf")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


# -- 1. RLHF pipeline ---------------------------------------------------------
def rlhf_pipeline():
    print("=== RLHF Pipeline ===")
    print()
    print("  Three stages:")
    print()
    print("  Stage 1: SFT (Supervised Fine-Tuning)")
    print("    - Collect human-written demonstrations")
    print("    - Fine-tune base LLM on these examples")
    print("    - Result: SFT model (better instruction following)")
    print()
    print("  Stage 2: Reward Model Training")
    print("    - Collect comparison data: (prompt, response_A, response_B, preference)")
    print("    - Train reward model R_theta to predict human preferences")
    print("    - R_theta(x, y) -> scalar score; higher = more preferred")
    print()
    print("  Stage 3: RL Fine-Tuning with PPO")
    print("    - Use R_theta as reward signal")
    print("    - Optimise policy pi_phi to maximise expected reward")
    print("    - KL penalty prevents policy from diverging from SFT model")
    print()
    print("  Full objective:")
    print("    L_RLHF = E_t[R_theta(x, y_t)] - beta · KL[pi_phi || pi_SFT]")


# -- 2. Reward model training --------------------------------------------------
def reward_model():
    print("\n=== Reward Model Training ===")
    print()
    print("  Loss: L_RM = -E[(x,y_w,y_l)] log sigma(R(x,y_w) - R(x,y_l))")
    print("  (Bradley-Terry preference model)")
    print()

    rng = np.random.default_rng(0)
    n_pairs = 200

    # Simulate reward model training
    def reward_model_forward(x, w):
        return x @ w

    d = 8
    w = rng.normal(0, 0.1, d)   # reward model weights

    losses = []
    for step in range(100):
        # Simulate chosen/rejected feature vectors
        x_chosen  = rng.normal(0.5, 1, (32, d))   # slightly better on average
        x_rejected = rng.normal(-0.5, 1, (32, d))

        r_chosen   = reward_model_forward(x_chosen, w)
        r_rejected = reward_model_forward(x_rejected, w)

        margin = r_chosen - r_rejected
        loss   = -np.log(sigmoid(margin) + 1e-9).mean()
        losses.append(loss)

        # Gradient (analytical for Bradley-Terry)
        grad_margin = sigmoid(margin) - 1   # (32,)
        grad_w = (x_chosen - x_rejected).T @ grad_margin / 32
        w -= 0.01 * grad_w

    # Evaluate
    x_test_c = rng.normal(0.5, 1, (100, d))
    x_test_r  = rng.normal(-0.5, 1, (100, d))
    acc = (reward_model_forward(x_test_c, w) > reward_model_forward(x_test_r, w)).mean()

    print(f"  Training: {len(losses)} steps")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss:   {losses[-1]:.4f}")
    print(f"  Test accuracy (chosen > rejected): {acc:.3f}")

    print()
    print("  Reward model design choices:")
    choices = [
        ("Architecture",  "Same as SFT model, replace LM head with scalar head"),
        ("Initialisation","SFT model weights; same size as policy"),
        ("Data scale",    "50k–1M preference pairs; quality > quantity"),
        ("Calibration",   "Normalise rewards; mean-0 std-1 per batch"),
        ("OOD handling",  "Reward model fails on novel inputs; KL keeps policy close"),
    ]
    for c, d in choices:
        print(f"  {c:<16} {d}")


# -- 3. PPO for LLMs -----------------------------------------------------------
def ppo_for_llms():
    print("\n=== PPO for LLM Fine-Tuning ===")
    print()
    print("  PPO (Proximal Policy Optimisation) adapts RL for LLMs:")
    print()
    print("  Key components:")
    components = [
        ("Policy (pi)",         "The LLM being trained; generates tokens"),
        ("Reference policy",   "Frozen SFT model; KL anchor"),
        ("Reward model (R_theta)", "Scores full response; scalar at EOS"),
        ("Value model (V_phi)",  "Estimates expected return; same size as policy"),
        ("KL penalty",         "r_t = R_theta(y) - beta·Sigma_t log pi(y_t|y_<t,x) / pi_ref"),
    ]
    for c, d in components:
        print(f"  {c:<22} {d}")

    print()
    print("  PPO clip objective:")
    print("    L_PPO = E[min(r_t·A_t, clip(r_t, 1-epsilon, 1+epsilon)·A_t)]")
    print("    r_t = pi_new(a_t|s_t) / pi_old(a_t|s_t)  (probability ratio)")
    print("    A_t = advantage estimate (GAE)")
    print()
    print("  Practical challenges:")
    challenges = [
        ("Credit assignment",  "Only 1 reward at end of sequence"),
        ("PPO instability",    "LLM PPO tricky; many hyperparams"),
        ("Reward hacking",     "Policy finds reward model loopholes"),
        ("KL budget",         "beta=0.1–0.5; balance reward vs drift"),
        ("Memory",            "4 models: policy, ref, reward, value"),
    ]
    for c, d in challenges:
        print(f"  {c:<20} {d}")


# -- 4. RLHF alternatives ------------------------------------------------------
def rlhf_alternatives():
    print("\n=== RLHF Alternatives ===")
    print()
    alternatives = [
        ("DPO",             "Direct Preference Optimisation; no RL; close-form loss"),
        ("ORPO",            "Odds ratio; no reference model; simpler than DPO"),
        ("RLOO",            "REINFORCE Leave-One-Out; online RL; no value model"),
        ("GRPO",            "Group Relative Policy Optimisation; DeepSeek-R1"),
        ("Constitutional AI","Self-critique and revision; no human preferences needed"),
        ("RLAIF",           "RL from AI feedback; Claude as labeller; Llama 2"),
        ("Iterative DPO",   "Online data generation; sample then DPO; improves"),
        ("RewardBench",     "Evaluator of reward models; standardised benchmark"),
    ]
    print(f"  {'Method':<22} {'Description'}")
    for m, d in alternatives:
        print(f"  {m:<22} {d}")


if __name__ == "__main__":
    rlhf_pipeline()
    reward_model()
    ppo_for_llms()
    rlhf_alternatives()
