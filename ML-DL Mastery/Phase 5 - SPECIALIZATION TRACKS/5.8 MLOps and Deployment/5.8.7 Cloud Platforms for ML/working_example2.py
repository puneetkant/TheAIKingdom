"""
Working Example 2: Cloud Platforms — cost estimation, region selection, managed services
=========================================================================================
Compares cloud ML service cost models and builds a budget estimator.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

# Approximate pricing (USD/hr, illustrative)
INSTANCE_PRICES = {
    "aws_p3.2xlarge":   3.06,    # 1 V100
    "aws_p4d.24xlarge": 32.77,   # 8 A100
    "gcp_a2-highgpu-1g": 3.67,   # 1 A100
    "azure_nc6s_v3":    3.06,    # 1 V100
    "aws_g4dn.xlarge":  0.526,   # 1 T4 (inference)
}

def estimate_training_cost(instance, hours, spot_discount=0.0):
    price = INSTANCE_PRICES.get(instance, 1.0) * (1 - spot_discount)
    return price * hours

def estimate_inference_cost(requests_per_day, latency_ms, replicas=1, instance="aws_g4dn.xlarge"):
    """
    Estimate based on: need N replicas to handle requests within latency SLA.
    Assume each instance handles 1000ms/latency_ms requests/s.
    """
    rps = requests_per_day / 86400
    capacity_per_instance = 1000 / latency_ms
    needed_instances = max(replicas, int(np.ceil(rps / capacity_per_instance)))
    daily_hours = 24 * needed_instances
    return INSTANCE_PRICES.get(instance, 0.5) * daily_hours, needed_instances

def demo():
    print("=== Cloud Platforms for ML ===\n")
    print("  Training Cost Estimates (30h run):")
    for inst, price in INSTANCE_PRICES.items():
        cost = estimate_training_cost(inst, 30)
        spot = estimate_training_cost(inst, 30, spot_discount=0.7)
        print(f"    {inst:30s}  on-demand=${cost:7.2f}  spot~=${spot:6.2f}")

    print("\n  Inference Cost (1M req/day, 50ms latency):")
    daily_cost, n_instances = estimate_inference_cost(1_000_000, 50, instance="aws_g4dn.xlarge")
    print(f"    Instances needed: {n_instances}  daily cost: ${daily_cost:.2f}  monthly: ${daily_cost*30:.0f}")

    # Cost vs training hours
    hours = np.arange(1, 101)
    costs = {
        "aws_p3.2xl (on-demand)": [estimate_training_cost("aws_p3.2xlarge", h) for h in hours],
        "aws_p3.2xl (spot 70%)":  [estimate_training_cost("aws_p3.2xlarge", h, 0.7) for h in hours],
        "gcp_a2-highgpu (on-demand)": [estimate_training_cost("gcp_a2-highgpu-1g", h) for h in hours],
    }
    plt.figure(figsize=(6, 3))
    for label, vals in costs.items():
        plt.plot(hours, vals, label=label)
    plt.xlabel("Training hours"); plt.ylabel("Cost (USD)")
    plt.title("Training Cost by Instance"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(OUTPUT / "cloud_platforms.png"); plt.close()
    print("\n  Saved cloud_platforms.png")

def demo_spot_strategy():
    """Show break-even analysis for spot vs on-demand instances."""
    print("\n=== Spot Instance Break-even Analysis ===")
    print(f"  {'Instance':30s}  {'On-demand':>10}  {'Spot(70%)':>10}  {'Break-even hrs':>15}")
    for inst, price in INSTANCE_PRICES.items():
        spot = price * 0.30
        # Break even: if interrupted and re-started, overhead = 15min per restart
        overhead_hrs = 0.25  # 15 min to re-checkpoint and restart
        # At what job length does spot savings outweigh restart risk?
        # Assume 10% interruption probability: E[loss] = 0.1 * overhead
        e_loss = 0.10 * overhead_hrs * price
        savings_per_hr = price - spot
        if savings_per_hr > 0:
            breakeven = e_loss / savings_per_hr
        else:
            breakeven = float("inf")
        print(f"  {inst:30s}  ${price:>9.3f}  ${spot:>9.3f}  {breakeven:>14.2f}h")


def demo_multi_cloud_comparison():
    """Compare equivalent GPU instances across cloud providers."""
    print("\n=== Multi-Cloud Comparison (1 V100 equivalent) ===")
    v100_instances = {
        "AWS p3.2xlarge":  3.06,
        "GCP n1-standard (V100)": 2.48,
        "Azure NC6s_v3":   3.06,
    }
    hours_per_month = 730
    print(f"  {'Provider':30s}  {'$/hr':>6}  {'$/month':>9}  {'Spot ~$/month':>14}")
    for provider, rate in v100_instances.items():
        monthly = rate * hours_per_month
        spot_monthly = rate * 0.30 * hours_per_month
        print(f"  {provider:30s}  {rate:>6.3f}  {monthly:>9.0f}  {spot_monthly:>14.0f}")


if __name__ == "__main__":
    demo()
    demo_spot_strategy()
    demo_multi_cloud_comparison()
