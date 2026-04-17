"""
Working Example 2: PyTorch Fundamentals — tensors, autograd, training loop
============================================================================
Complete end-to-end PyTorch workflow on California Housing.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install torch scikit-learn matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def demo_tensors():
    print("=== PyTorch Tensors and Autograd ===")
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    y = (x**2).sum()       # y = x0² + x1²
    y.backward()
    print(f"  x = {x.detach().numpy()}  y = {y.item():.1f}")
    print(f"  dy/dx = {x.grad.numpy()}  (expected: [4.0, 6.0])")

def demo_training():
    print("\n=== PyTorch MLP on California Housing ===")
    h = fetch_california_housing(); X = StandardScaler().fit_transform(h.data)
    X_tr, X_te, y_tr, y_te = train_test_split(X, h.target, test_size=0.2, random_state=42)

    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    Xe = torch.tensor(X_te, dtype=torch.float32)
    ye = torch.tensor(y_te, dtype=torch.float32).unsqueeze(1)

    model = nn.Sequential(
        nn.Linear(8, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 1),
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    losses = []
    for epoch in range(200):
        model.train()
        pred = model(Xt); loss = loss_fn(pred, yt)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(loss.item())
        if (epoch+1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(Xe), ye).item()
            print(f"  Epoch {epoch+1:3d}: train={loss.item():.4f}  val={val_loss:.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE"); ax.set_title("PyTorch MLP Training")
    plt.tight_layout(); plt.savefig(OUTPUT / "pytorch_training.png"); plt.close()
    print("  Saved pytorch_training.png")

if __name__ == "__main__":
    demo_tensors()
    demo_training()
