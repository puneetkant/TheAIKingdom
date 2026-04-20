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

def demo_custom_dataset():
    """Demonstrate PyTorch Dataset and DataLoader abstractions."""
    print("\n=== PyTorch Dataset and DataLoader ===")
    import torch.utils.data as data_utils

    h = fetch_california_housing(); X = StandardScaler().fit_transform(h.data)
    X_tr, X_te, y_tr, y_te = train_test_split(X, h.target, test_size=0.2, random_state=42)

    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    dataset = data_utils.TensorDataset(Xt, yt)
    loader  = data_utils.DataLoader(dataset, batch_size=64, shuffle=True)

    model = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 1))
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(3):   # 3 quick epochs for demo
        epoch_loss = 0.0
        for Xb, yb in loader:
            pred = model(Xb); loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
        n_batches = len(loader)
        print(f"  Epoch {epoch+1}: avg_batch_loss={epoch_loss/n_batches:.4f}  "
              f"({n_batches} batches of 64)")


def demo_gradient_clipping():
    """Show gradient clipping to prevent exploding gradients."""
    print("\n=== Gradient Clipping ===")
    h = fetch_california_housing(); X = StandardScaler().fit_transform(h.data)
    X_tr, _, y_tr, _ = train_test_split(X, h.target, test_size=0.2, random_state=42)
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)

    model = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 1))
    opt   = optim.SGD(model.parameters(), lr=0.5)   # deliberately high lr
    loss_fn = nn.MSELoss()

    for epoch in range(5):
        pred = model(Xt); loss = loss_fn(pred, yt)
        opt.zero_grad(); loss.backward()
        grad_norm_before = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norm_after  = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5
        opt.step()
        print(f"  Epoch {epoch+1}: grad_norm before={grad_norm_before:.4f}  after clip={grad_norm_after:.4f}")


if __name__ == "__main__":
    demo_tensors()
    demo_training()
    demo_custom_dataset()
    demo_gradient_clipping()
