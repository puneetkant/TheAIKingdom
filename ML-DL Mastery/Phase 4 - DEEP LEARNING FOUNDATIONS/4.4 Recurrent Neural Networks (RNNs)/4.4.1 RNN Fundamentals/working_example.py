"""
Working Example: RNN Fundamentals
Covers recurrent neurons, BPTT, vanishing gradients, sequence processing,
teacher forcing, and from-scratch Elman RNN implementation.
"""
import numpy as np
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_rnn_fundamentals")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def tanh(z):      return np.tanh(z)
def tanh_d(z):    return 1 - np.tanh(z)**2
def sigmoid(z):   return 1 / (1 + np.exp(-z.clip(-500, 500)))
def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# -- 1. Motivation -------------------------------------------------------------
def motivation():
    print("=== RNNs: Why Recurrence? ===")
    print("  Standard MLPs assume fixed-size independent inputs")
    print("  Sequences have variable length and temporal dependencies")
    print()
    print("  Applications of sequence modelling:")
    print("    Text generation, language modelling   (many->many)")
    print("    Sentiment analysis                    (many->one)")
    print("    Machine translation                   (many->many)")
    print("    Time series forecasting               (many->one or many)")
    print("    Speech recognition                    (many->many)")
    print()
    print("  Key idea: maintain a hidden state h_t that summarises past inputs")
    print("  h_t = f(W_hh · h_{t-1} + W_xh · x_t + b_h)")
    print("  y_t = g(W_hy · h_t + b_y)")


# -- 2. Elman RNN from scratch -------------------------------------------------
class ElmanRNN:
    """Simple Elman RNN: h_t = tanh(W_hh h_{t-1} + W_xh x_t + b_h)."""

    def __init__(self, input_size, hidden_size, output_size, rng=None):
        rng = rng or np.random.default_rng(0)
        scale = 0.01
        self.W_xh = rng.standard_normal((input_size, hidden_size)) * scale
        self.W_hh = rng.standard_normal((hidden_size, hidden_size)) * scale
        self.b_h  = np.zeros(hidden_size)
        self.W_hy = rng.standard_normal((hidden_size, output_size)) * scale
        self.b_y  = np.zeros(output_size)
        self.hidden_size = hidden_size

    def forward(self, X_seq, h0=None):
        """
        X_seq: (T, input_size)
        Returns ys (T, output_size), hs (T+1, hidden_size)
        """
        T = X_seq.shape[0]
        h  = np.zeros(self.hidden_size) if h0 is None else h0
        hs = [h.copy()]
        ys = []
        self.cache = {"xs": X_seq, "hs": []}
        for t in range(T):
            h = tanh(X_seq[t] @ self.W_xh + h @ self.W_hh + self.b_h)
            y = softmax(h @ self.W_hy + self.b_y)
            hs.append(h.copy())
            ys.append(y.copy())
            self.cache["hs"].append(h.copy())
        self.cache["hs_all"] = np.array([np.zeros(self.hidden_size)] + self.cache["hs"])
        return np.array(ys), np.array(hs)

    def bptt(self, X_seq, Y_seq, lr=0.01):
        """Backpropagation through time."""
        T   = X_seq.shape[0]
        ys, hs = self.forward(X_seq)
        hs_all = np.array([np.zeros(self.hidden_size)] + list(hs[:-1]))

        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h  = np.zeros_like(self.b_h)
        dW_hy = np.zeros_like(self.W_hy)
        db_y  = np.zeros_like(self.b_y)

        # Output gradient: cross-entropy + softmax
        dy = ys - Y_seq          # (T, output_size)
        dW_hy += hs[:-1].T @ dy
        db_y  += dy.sum(axis=0)

        dh_next = np.zeros(self.hidden_size)
        for t in reversed(range(T)):
            dh = dy[t] @ self.W_hy.T + dh_next
            dh_raw = dh * tanh_d(X_seq[t] @ self.W_xh + hs_all[t] @ self.W_hh + self.b_h)
            dW_xh += np.outer(X_seq[t], dh_raw)
            dW_hh += np.outer(hs_all[t], dh_raw)
            db_h  += dh_raw
            dh_next = dh_raw @ self.W_hh.T

        # Gradient clipping
        for dW in [dW_xh, dW_hh, dW_hy, db_h, db_y]:
            np.clip(dW, -5, 5, out=dW)

        self.W_xh -= lr * dW_xh
        self.W_hh -= lr * dW_hh
        self.b_h  -= lr * db_h
        self.W_hy -= lr * dW_hy
        self.b_y  -= lr * db_y

        loss = -np.mean(Y_seq * np.log(ys.clip(1e-9)))
        return loss


def rnn_forward_demo():
    print("\n=== Elman RNN Forward Pass ===")
    rng = np.random.default_rng(42)
    T, input_size, hidden_size, output_size = 5, 3, 8, 2
    rnn = ElmanRNN(input_size, hidden_size, output_size, rng)
    X   = rng.standard_normal((T, input_size))
    ys, hs = rnn.forward(X)
    print(f"  Sequence length T = {T}")
    print(f"  Input size:   {input_size}")
    print(f"  Hidden size:  {hidden_size}")
    print(f"  Output size:  {output_size}")
    print(f"  X shape:  {X.shape}")
    print(f"  ys shape: {ys.shape}  (one output per time step)")
    print(f"  hs shape: {hs.shape}  (hidden state at each step)")
    for t in range(T):
        print(f"  t={t}: y={ys[t].round(3)}  h_norm={np.linalg.norm(hs[t]):.4f}")


# -- 3. Sequence types ---------------------------------------------------------
def sequence_types():
    print("\n=== RNN Sequence Types ===")
    types = [
        ("One->One",   "Single input -> single output",    "MLPs, image classification"),
        ("One->Many",  "Single input -> sequence output",  "Image captioning"),
        ("Many->One",  "Sequence input -> single output",  "Sentiment analysis"),
        ("Many->Many", "Same-length seq-to-seq",          "POS tagging, NER"),
        ("Enc->Dec",   "Variable len seq-to-seq",         "Machine translation"),
    ]
    print(f"  {'Type':<12} {'Description':<36} {'Example'}")
    for t, d, e in types:
        print(f"  {t:<12} {d:<36} {e}")


# -- 4. Vanishing gradient in RNNs ---------------------------------------------
def vanishing_gradient_rnn():
    print("\n=== Vanishing Gradient in RNNs ===")
    print("  During BPTT: deltaL/deltah_1 = deltaL/deltah_T · Pi_{t=2}^{T} (W_hh · diag(tanh'(z_t)))")
    print("  If spectral radius of W_hh < 1 -> gradients vanish exponentially")
    print("  If spectral radius of W_hh > 1 -> gradients explode")
    print()

    rng = np.random.default_rng(5)
    T = 50
    for scale, label in [(0.5, "Small W_hh (vanishing)"),
                          (1.0, "Unit W_hh"),
                          (1.1, "Large W_hh (exploding)")]:
        W = rng.standard_normal((8, 8)) * scale / np.sqrt(8)
        grad = np.eye(8)
        norms = []
        for _ in range(T):
            # tanh' ~= 0.5 on average; include this factor
            grad = 0.5 * W.T @ grad
            norms.append(np.linalg.norm(grad))
        print(f"  {label:<35}: norm after {T} steps = {norms[-1]:.4e}")


# -- 5. BPTT training demo -----------------------------------------------------
def bptt_training():
    print("\n=== BPTT Training: Sequence Classification ===")
    print("  Task: predict last element of sequence (parity of binary string)")

    rng  = np.random.default_rng(7)
    T    = 6
    rnn  = ElmanRNN(input_size=1, hidden_size=16, output_size=2, rng=rng)

    def make_batch(n, T, rng):
        xs = rng.integers(0, 2, (n, T, 1)).astype(float)
        ys = (xs[:, :, 0].sum(axis=1) % 2).astype(int)  # parity
        return xs, ys

    n_epochs = 100
    losses   = []
    for ep in range(n_epochs):
        Xb, yb = make_batch(32, T, rng)
        ep_loss = []
        for i in range(32):
            # Many-to-one: only last output matters
            Y_one_hot  = np.zeros((T, 2))
            Y_one_hot[-1, yb[i]] = 1.0
            loss = rnn.bptt(Xb[i], Y_one_hot, lr=0.05)
            ep_loss.append(loss)
        losses.append(np.mean(ep_loss))

    print(f"  Start loss: {losses[0]:.4f}")
    print(f"  End loss:   {losses[-1]:.4f}")

    # Evaluate
    Xtest, ytest = make_batch(200, T, rng)
    correct = 0
    for i in range(200):
        ys, _ = rnn.forward(Xtest[i])
        pred  = ys[-1].argmax()
        correct += int(pred == ytest[i])
    print(f"  Test accuracy: {correct/200:.4f}")

    # Plot training curve
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses, lw=2)
    ax.set(xlabel="Epoch", ylabel="Loss", title="RNN BPTT Training (Parity Task)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "rnn_training.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"  Training curve saved: {path}")


# -- 6. Teacher forcing --------------------------------------------------------
def teacher_forcing():
    print("\n=== Teacher Forcing ===")
    print("  During training: feed ground-truth y_{t-1} as next input")
    print("  During inference: feed model's own prediction y_{t-1}")
    print()
    print("  Pros: faster, more stable training; prevents error accumulation")
    print("  Cons: exposure bias — model never sees its own errors during training")
    print()
    print("  Scheduled sampling (Bengio 2015):")
    print("    epsilon_t = probability of using ground truth")
    print("    Start with epsilon=1, anneal to epsilon=0 as training progresses")
    print("    Bridges gap between teacher-forced training and free-running inference")


# -- 7. Truncated BPTT ---------------------------------------------------------
def truncated_bptt():
    print("\n=== Truncated BPTT ===")
    print("  Full BPTT on long sequences (T=1000) is expensive and unstable")
    print("  Truncated BPTT: propagate gradients only k steps back")
    print()
    print("  Algorithm:")
    print("  • Process sequence in chunks of size k (e.g. k=35 for language models)")
    print("  • Forward pass through chunk using last hidden state")
    print("  • Backward pass only through that chunk")
    print("  • Detach hidden state from graph between chunks")
    print()
    for k in [5, 10, 20, 50]:
        T = 200
        n_chunks = T // k
        print(f"  T={T}, k={k}: {n_chunks} chunks per sequence")


if __name__ == "__main__":
    motivation()
    rnn_forward_demo()
    sequence_types()
    vanishing_gradient_rnn()
    bptt_training()
    teacher_forcing()
    truncated_bptt()
