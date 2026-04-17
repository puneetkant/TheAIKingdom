"""
Working Example 2: Classical Computer Vision — edge detection, Harris corners, HOG features
=============================================================================================
Implements Sobel edge detection and simple feature extraction using numpy.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_digits
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def sobel(img):
    """Sobel edge detection on 2D grayscale image."""
    Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=float)
    Ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=float)
    def conv2d(img, K):
        H, W = img.shape; kH, kW = K.shape
        pad = kH // 2
        out = np.zeros_like(img)
        p = np.pad(img, pad)
        for i in range(H):
            for j in range(W):
                out[i,j] = (p[i:i+kH, j:j+kW] * K).sum()
        return out
    Gx = conv2d(img, Kx); Gy = conv2d(img, Ky)
    return np.sqrt(Gx**2 + Gy**2)

def demo():
    print("=== Classical Computer Vision ===")
    digits = load_digits()
    # Use a digit image (8×8)
    img = digits.images[0]  # 8×8 grayscale
    edges = sobel(img)
    print(f"  Digit image shape: {img.shape}  Edge max: {edges.max():.2f}")

    # HOG-like feature: histogram of gradients per block
    def hog_features(images, n_bins=9):
        feats = []
        for im in images:
            Kx = np.array([[-1,0,1]]); Ky = Kx.T
            pad_im = np.pad(im, 1)
            Gx = np.array([[((pad_im[i:i+1, j:j+3])*Kx).sum() for j in range(im.shape[1])] for i in range(im.shape[0])])
            Gy = np.array([[((pad_im[i:i+3, j:j+1])*Ky).sum() for j in range(im.shape[1])] for i in range(im.shape[0])])
            mag = np.sqrt(Gx**2 + Gy**2); ang = np.arctan2(Gy, Gx+1e-9) * 180 / np.pi % 180
            hist, _ = np.histogram(ang.ravel(), bins=n_bins, range=(0,180), weights=mag.ravel())
            feats.append(hist / (hist.sum() + 1e-9))
        return np.array(feats)

    # Classify digits with HOG features vs raw pixels
    X_hog = hog_features(digits.images)
    X_raw = digits.data / 16.0
    X_tr_h, X_te_h, y_tr, y_te = train_test_split(X_hog, digits.target, test_size=0.2, random_state=42)
    X_tr_r, X_te_r, _, _ = train_test_split(X_raw, digits.target, test_size=0.2, random_state=42)

    acc_hog = accuracy_score(y_te, SVC(kernel="rbf").fit(X_tr_h, y_tr).predict(X_te_h))
    acc_raw = accuracy_score(y_te, SVC(kernel="rbf").fit(X_tr_r, y_tr).predict(X_te_r))
    print(f"  HOG features ({X_hog.shape[1]}d): {acc_hog:.4f}")
    print(f"  Raw pixels  ({X_raw.shape[1]}d): {acc_raw:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("Original digit")
    axes[1].imshow(edges, cmap="hot"); axes[1].set_title("Sobel edges")
    axes[2].bar(range(9), X_hog[0]); axes[2].set_title("HOG histogram")
    plt.tight_layout(); plt.savefig(OUTPUT / "classical_cv.png"); plt.close()
    print("  Saved classical_cv.png")

if __name__ == "__main__":
    demo()
