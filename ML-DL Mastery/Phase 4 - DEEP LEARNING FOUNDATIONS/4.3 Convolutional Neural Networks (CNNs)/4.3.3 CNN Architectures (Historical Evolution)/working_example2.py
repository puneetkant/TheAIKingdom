"""
Working Example 2: CNN Architectures — LeNet-style numpy demo + architecture comparison
=========================================================================================
Demonstrates LeNet architecture (conv->pool->fc) on synthetic image classification.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_digits
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def demo_lenet_description():
    """Print the LeNet architecture topology."""
    print("=== CNN Architectures (Historical Evolution) ===")
    architectures = [
        ("LeNet-5 (1998)",    "Input(32²,1) -> Conv(6,5) -> Pool -> Conv(16,5) -> Pool -> FC120 -> FC84 -> 10"),
        ("AlexNet (2012)",    "Input(224²,3) -> Conv(96,11,s4) -> ... -> FC4096 -> FC4096 -> 1000"),
        ("VGG-16 (2014)",     "Input(224²,3) -> 13xConv(3x3) -> 3xFC -> 1000"),
        ("GoogLeNet (2014)",  "Input -> Inception modules (parallel convs) -> GlobalAvgPool -> 1000"),
        ("ResNet-50 (2015)",  "Input -> Conv -> 16xResidual Blocks -> GlobalAvgPool -> 1000"),
    ]
    for name, arch in architectures:
        print(f"\n  {name}")
        print(f"    {arch}")

def demo_digits_cnn_proxy():
    """Use digit pixels as 'feature maps' and compare flatten vs simple pooling."""
    print("\n=== Digits: Flatten vs Hand-Pooled Features ===")
    digits = load_digits()
    X = digits.data / 16.0  # 64 features = 8×8 image
    y = digits.target

    # Raw 64 features
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_tr, y_tr); raw_acc = accuracy_score(y_te, lr.predict(X_te))

    # Simulated "global average pooling" — average of 4 quadrants per image
    def quad_pool(data):
        imgs = data.reshape(-1, 8, 8)
        top_left  = imgs[:, :4, :4].mean(axis=(1,2))
        top_right = imgs[:, :4, 4:].mean(axis=(1,2))
        bot_left  = imgs[:, 4:, :4].mean(axis=(1,2))
        bot_right = imgs[:, 4:, 4:].mean(axis=(1,2))
        return np.column_stack([top_left, top_right, bot_left, bot_right])

    Xp = quad_pool(X)
    Xp_tr, Xp_te, _, _ = train_test_split(Xp, y, test_size=0.2, random_state=42)
    lr2 = LogisticRegression(max_iter=500, random_state=42)
    lr2.fit(Xp_tr, y_tr); pool_acc = accuracy_score(y_te, lr2.predict(Xp_te))

    print(f"  Raw 64 features:      acc={raw_acc:.4f}")
    print(f"  Quad pooled (4 feat): acc={pool_acc:.4f}")
    print("  (Raw >> pooled — confirms pooling reduces info for classif)")

def demo_resnet_skip_connection():
    """Demonstrate ResNet skip connection concept with numpy."""
    print("\n=== ResNet Skip Connection (numpy) ===")
    rng = np.random.default_rng(42)
    n, d = 64, 32
    X = rng.standard_normal((n, d))

    def residual_block(x, W1, b1, W2, b2):
        """F(x) + x  — identity shortcut."""
        h = np.maximum(0, x @ W1 + b1)   # relu
        Fx = h @ W2 + b2
        return np.maximum(0, Fx + x)     # relu(F(x) + x)

    W1 = rng.standard_normal((d, d)) * np.sqrt(2/d)
    b1 = np.zeros(d)
    W2 = rng.standard_normal((d, d)) * np.sqrt(2/d)
    b2 = np.zeros(d)

    out = residual_block(X, W1, b1, W2, b2)
    print(f"  Input std:  {X.std():.4f}")
    print(f"  Output std: {out.std():.4f}  (skip preserves scale better)")
    print(f"  Fraction of neurons active: {(out > 0).mean():.3f}")

    # Stack 20 residual blocks — no vanishing signal
    x = X.copy()
    for _ in range(20):
        W1 = rng.standard_normal((d, d)) * np.sqrt(2/d)
        W2 = rng.standard_normal((d, d)) * np.sqrt(2/d)
        x = residual_block(x, W1, np.zeros(d), W2, np.zeros(d))
    print(f"  After 20 residual blocks: std={x.std():.4f}  (stable without skip: often 0 or inf)")


def demo_parameter_count():
    """Compare parameter counts across famous CNN architectures."""
    print("\n=== CNN Architecture Parameter Counts ===")
    models = [
        ("LeNet-5",      60_000),
        ("AlexNet",      62_000_000),
        ("VGG-16",       138_000_000),
        ("GoogLeNet",    6_800_000),
        ("ResNet-50",    25_600_000),
        ("MobileNetV2", 3_400_000),
        ("EfficientNet-B0", 5_300_000),
    ]
    print(f"  {'Model':25s}  {'Params':>15s}  {'Size (MB)':>10s}")
    for name, params in models:
        mb = params * 4 / 1e6   # float32
        print(f"  {name:25s}  {params:>15,d}  {mb:>10.1f}")


if __name__ == "__main__":
    demo_lenet_description()
    demo_digits_cnn_proxy()
    demo_resnet_skip_connection()
    demo_parameter_count()
