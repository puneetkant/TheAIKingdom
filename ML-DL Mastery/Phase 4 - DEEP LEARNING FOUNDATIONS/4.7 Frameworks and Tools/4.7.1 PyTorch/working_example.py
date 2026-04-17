"""
Working Example: PyTorch
Covers tensors, autograd, nn.Module, training loop, DataLoader,
GPU usage, and common patterns — with graceful fallback if not installed.
"""

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os


# ── 1. Tensors ────────────────────────────────────────────────────────────────
def tensor_basics():
    print("=== PyTorch Tensors ===")
    if not HAS_TORCH:
        print("  [PyTorch not installed — showing code patterns only]")
        print()
        print("  import torch")
        print("  x = torch.tensor([1.0, 2.0, 3.0])         # from list")
        print("  x = torch.zeros(3, 4)                      # zeros tensor")
        print("  x = torch.randn(2, 3)                      # normal random")
        print("  x = torch.arange(0, 10, step=2)            # range")
        print()
        print("  # Operations")
        print("  y = x + 1; z = x @ x.T; w = x.mean(dim=1)")
        print("  # Reshaping")
        print("  x.view(6); x.reshape(1, -1); x.squeeze(); x.unsqueeze(0)")
        print("  # Device")
        print("  x.to('cuda'); x.cuda(); x.cpu()")
        return

    x = torch.tensor([1.0, 2.0, 3.0])
    z = torch.zeros(3, 4)
    r = torch.randn(2, 3)
    print(f"  tensor:   {x}")
    print(f"  zeros:    {z.shape}")
    print(f"  randn:    {r.shape}  dtype={r.dtype}")
    print(f"  x+1:      {x+1}")
    print(f"  x·x:      {(x*x).sum().item():.0f}")
    print(f"  reshape:  {x.unsqueeze(0).shape}")
    print(f"  numpy:    {x.numpy()}")
    print(f"  CUDA available: {torch.cuda.is_available()}")


# ── 2. Autograd ───────────────────────────────────────────────────────────────
def autograd_demo():
    print("\n=== Autograd — Automatic Differentiation ===")
    if not HAS_TORCH:
        print("  [PyTorch not installed — code pattern below]")
        print()
        print("  x = torch.tensor(3.0, requires_grad=True)")
        print("  y = x**2 + 2*x + 1              # y = (x+1)²")
        print("  y.backward()                     # compute gradients")
        print("  print(x.grad)                    # dy/dx = 2(x+1) = 8")
        print()
        print("  # No-grad context (inference mode):")
        print("  with torch.no_grad():")
        print("      y = model(x)")
        return

    x = torch.tensor(3.0, requires_grad=True)
    y = x**2 + 2*x + 1       # y = (x+1)²
    y.backward()
    print(f"  x = 3.0")
    print(f"  y = x² + 2x + 1 = {y.item()}")
    print(f"  dy/dx = 2x + 2 = {x.grad.item()}  (expected 8.0)")

    # Chain rule example
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    loss = (x**2).sum()
    loss.backward()
    print(f"\n  x = {x.data}")
    print(f"  loss = Σx²")
    print(f"  dloss/dx = 2x = {x.grad}")

    # No-grad
    with torch.no_grad():
        val = x**2 + 1
    print(f"\n  no_grad context: val.requires_grad = {val.requires_grad}")


# ── 3. nn.Module ──────────────────────────────────────────────────────────────
def nn_module_demo():
    print("\n=== nn.Module — Building Neural Networks ===")
    if not HAS_TORCH:
        print("  [PyTorch not installed — code pattern below]")
        print()
        print("  class MLP(nn.Module):")
        print("      def __init__(self, in_dim, hidden, out_dim):")
        print("          super().__init__()")
        print("          self.net = nn.Sequential(")
        print("              nn.Linear(in_dim, hidden),")
        print("              nn.ReLU(),")
        print("              nn.Dropout(0.3),")
        print("              nn.Linear(hidden, out_dim))")
        print()
        print("      def forward(self, x):")
        print("          return self.net(x)")
        print()
        print("  model = MLP(64, 128, 10)")
        print("  print(model)  # layer summary")
        print("  n_params = sum(p.numel() for p in model.parameters())")
        return

    class MLP(nn.Module):
        def __init__(self, in_dim, hidden, out_dim, dropout=0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, out_dim)
            )
        def forward(self, x):
            return self.net(x)

    model = MLP(64, 128, 10)
    print(model)
    n = sum(p.numel() for p in model.parameters())
    print(f"\n  Total parameters: {n:,}")
    x = torch.randn(8, 64)
    y = model(x)
    print(f"  Forward: {x.shape} → {y.shape}")


# ── 4. Training loop ──────────────────────────────────────────────────────────
def training_loop():
    print("\n=== Standard PyTorch Training Loop ===")
    if not HAS_TORCH:
        print("  [PyTorch not installed — code pattern below]")
        print()
        training_loop_pseudocode()
        return

    # Data
    rng = np.random.default_rng(0)
    X, y = make_classification(n_samples=1000, n_features=20, random_state=0)
    X = StandardScaler().fit_transform(X)
    Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.2, random_state=0)

    Xtr_t = torch.FloatTensor(Xtr); ytr_t = torch.LongTensor(ytr)
    Xts_t = torch.FloatTensor(Xts); yts_t = torch.LongTensor(yts)
    dataset = TensorDataset(Xtr_t, ytr_t)
    loader  = DataLoader(dataset, batch_size=32, shuffle=True)

    model     = nn.Sequential(
        nn.Linear(20, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 2)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    for epoch in range(30):
        model.train()
        ep_loss = 0
        for Xb, yb in loader:
            optimizer.zero_grad()
            logits = model(Xb)
            loss   = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss += loss.item() * len(Xb)
        scheduler.step()

    model.eval()
    with torch.no_grad():
        logits = model(Xts_t)
        preds  = logits.argmax(dim=1)
        acc    = (preds == yts_t).float().mean().item()
    print(f"  Test accuracy: {acc:.4f}")
    print(f"  Final LR: {scheduler.get_last_lr()[0]:.6f}")


def training_loop_pseudocode():
    code = """
  model = MyModel()
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=1e-3)

  for epoch in range(n_epochs):
      model.train()
      for X_batch, y_batch in train_loader:
          optimizer.zero_grad()          # clear old gradients
          logits = model(X_batch)        # forward
          loss   = criterion(logits, y_batch)
          loss.backward()                # backward
          nn.utils.clip_grad_norm_(model.parameters(), 1.0)
          optimizer.step()               # update weights

      model.eval()
      with torch.no_grad():
          val_loss = evaluate(model, val_loader)
      scheduler.step(val_loss)

      # Save checkpoint
      if val_loss < best_val_loss:
          best_val_loss = val_loss
          torch.save(model.state_dict(), 'best_model.pt')

  # Load best model
  model.load_state_dict(torch.load('best_model.pt'))
    """
    print(code)


# ── 5. GPU usage ──────────────────────────────────────────────────────────────
def gpu_patterns():
    print("\n=== GPU Usage Patterns ===")
    print("""
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model  = model.to(device)

  # Training
  for X, y in loader:
      X, y = X.to(device), y.to(device)
      loss = criterion(model(X), y)
      ...

  # DataParallel (multi-GPU, single machine)
  if torch.cuda.device_count() > 1:
      model = nn.DataParallel(model)

  # Mixed precision training
  scaler = torch.cuda.amp.GradScaler()
  with torch.autocast(device_type='cuda'):
      loss = criterion(model(X), y)
  scaler.scale(loss).backward()
  scaler.step(optimizer)
  scaler.update()
    """)


# ── 6. Common patterns ────────────────────────────────────────────────────────
def common_patterns():
    print("=== Common PyTorch Patterns ===")
    patterns = [
        ("Save / load model",   "torch.save(model.state_dict(), 'model.pt')"),
        ("",                    "model.load_state_dict(torch.load('model.pt'))"),
        ("Freeze layers",       "for p in model.encoder.parameters(): p.requires_grad=False"),
        ("Transfer learning",   "model = torchvision.models.resnet50(pretrained=True)"),
        ("Inference mode",      "with torch.no_grad(): y = model(x)"),
        ("Gradient clipping",   "nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)"),
        ("LR Scheduler",        "scheduler = CosineAnnealingLR(opt, T_max=100)"),
        ("Custom loss",         "class MyLoss(nn.Module): def forward(self, y, t): ..."),
        ("Dataset class",       "class MyDS(Dataset): def __getitem__(self, i): ..."),
        ("Reproducibility",     "torch.manual_seed(42); np.random.seed(42)"),
    ]
    for name, code in patterns:
        if name:
            print(f"\n  {name}:")
        print(f"    {code}")


if __name__ == "__main__":
    tensor_basics()
    autograd_demo()
    nn_module_demo()
    training_loop()
    gpu_patterns()
    common_patterns()
