"""
Working Example: CNN Architectures (Historical Evolution)
Covers LeNet-5, AlexNet, VGG, GoogLeNet/Inception, ResNet, DenseNet,
MobileNet, and EfficientNet design principles.
"""
import numpy as np


# -- 1. LeNet-5 (LeCun 1998) --------------------------------------------------
def lenet5():
    print("=== LeNet-5 (LeCun et al., 1998) ===")
    print("  First successful CNN; designed for digit recognition (MNIST)")
    print()
    layers = [
        ("Input",    "32×32×1",   "Grayscale image"),
        ("Conv C1",  "28×28×6",   "6 kernels 5×5, stride 1"),
        ("Pool S2",  "14×14×6",   "Avg pool 2×2, stride 2"),
        ("Conv C3",  "10×10×16",  "16 kernels 5×5"),
        ("Pool S4",  "5×5×16",    "Avg pool 2×2, stride 2"),
        ("Conv C5",  "1×1×120",   "120 kernels 5×5 (=FC layer)"),
        ("FC F6",    "84",        "Dense, tanh"),
        ("Output",   "10",        "Softmax (Gaussian RBF originally)"),
    ]
    print(f"  {'Layer':<12} {'Shape':<14} {'Details'}")
    for name, shape, detail in layers:
        print(f"  {name:<12} {shape:<14} {detail}")
    params = 6*5*5 + 16*5*5 + 120*5*5 + 84*120 + 10*84
    print(f"\n  Total parameters: ~60,000  (vs modern millions)")
    print(f"  Activation: tanh   Pooling: average   Loss: MSE")


# -- 2. AlexNet (Krizhevsky 2012) ---------------------------------------------
def alexnet():
    print("\n=== AlexNet (Krizhevsky et al., 2012) ===")
    print("  Won ImageNet ILSVRC 2012 (top-5 error 15.3% vs 26.2% runner-up)")
    print("  Trained on 2 GTX 580 GPUs with model parallelism")
    print()
    innovations = [
        ("ReLU activations",        "Replaced tanh; 6× faster convergence"),
        ("GPU training",            "First large-scale GPU CNN"),
        ("Dropout (p=0.5)",         "Used in FC layers; reduced overfitting"),
        ("Data augmentation",       "Random crops, flips, colour jitter"),
        ("Local Response Norm",     "Lateral inhibition (later superseded by BN)"),
        ("Overlapping max pooling", "3×3 pool stride 2 (not 2×2 stride 2)"),
    ]
    for name, desc in innovations:
        print(f"  [OK] {name:<30}: {desc}")
    print()
    layers = [
        ("Input",      "227×227×3"),
        ("Conv1 96k",  "55×55×96    k=11 s=4"),
        ("Pool1",      "27×27×96    3×3 s=2"),
        ("Conv2 256k", "27×27×256   k=5 pad=2"),
        ("Pool2",      "13×13×256   3×3 s=2"),
        ("Conv3 384k", "13×13×384   k=3 pad=1"),
        ("Conv4 384k", "13×13×384   k=3 pad=1"),
        ("Conv5 256k", "13×13×256   k=3 pad=1"),
        ("Pool5",      "6×6×256     3×3 s=2"),
        ("FC6 4096",   "4096        Dropout"),
        ("FC7 4096",   "4096        Dropout"),
        ("Output",     "1000        Softmax"),
    ]
    print(f"  {'Layer':<14} {'Shape + info'}")
    for name, shape in layers:
        print(f"  {name:<14} {shape}")
    print(f"\n  Total parameters: ~62 million")


# -- 3. VGG (Simonyan 2014) ---------------------------------------------------
def vggnet():
    print("\n=== VGGNet (Simonyan & Zisserman, 2014) ===")
    print("  Key insight: depth matters — use small 3×3 kernels stacked")
    print("  Two 3x3 convs ≡ one 5x5 conv in RF, but fewer params and more nonlinearity")
    print("  Three 3x3 convs ≡ one 7x7 conv")
    print()

    vgg16_blocks = [
        ("Block 1", ["Conv 64, 3×3, pad=1"]*2,      "224×224×64"),
        ("Pool 1",  ["MaxPool 2×2 s=2"],              "112×112×64"),
        ("Block 2", ["Conv 128, 3×3, pad=1"]*2,      "112×112×128"),
        ("Pool 2",  ["MaxPool 2×2 s=2"],              "56×56×128"),
        ("Block 3", ["Conv 256, 3×3, pad=1"]*3,      "56×56×256"),
        ("Pool 3",  ["MaxPool 2×2 s=2"],              "28×28×256"),
        ("Block 4", ["Conv 512, 3×3, pad=1"]*3,      "28×28×512"),
        ("Pool 4",  ["MaxPool 2×2 s=2"],              "14×14×512"),
        ("Block 5", ["Conv 512, 3×3, pad=1"]*3,      "14×14×512"),
        ("Pool 5",  ["MaxPool 2×2 s=2"],              "7×7×512"),
        ("FC",      ["FC 4096","FC 4096","FC 1000"],  "1000 classes"),
    ]
    print(f"  {'Block':<12} {'Layers':<40} {'Output'}")
    for name, layers, output in vgg16_blocks:
        print(f"  {name:<12} {', '.join(layers):<40} {output}")

    # VGG16 params (approx)
    fc_params = 7*7*512*4096 + 4096*4096 + 4096*1000
    conv_params = 2*(3*3*3*64 + 3*3*64*64) + 2*(3*3*64*128 + 3*3*128*128) + \
                  3*(3*3*128*256 + 3*3*256*256 + 3*3*256*256) + \
                  6*(3*3*256*512 + 3*3*512*512)
    print(f"\n  Conv params:  ~{conv_params/1e6:.0f}M")
    print(f"  FC params:    ~{fc_params/1e6:.0f}M (dominate!)")
    print(f"  Total VGG16:  ~138M parameters")


# -- 4. GoogLeNet / Inception (2014) -------------------------------------------
def googlenet_inception():
    print("\n=== GoogLeNet / Inception v1 (Szegedy et al., 2014) ===")
    print("  Winner of ILSVRC 2014 (22 layers, only 5M params vs VGG 138M)")
    print()
    print("  Inception Module: parallel branches at multiple scales")
    print()
    print("  +-----------------------------------------------------+")
    print("  |  Input                                               |")
    print("  |    |         |              |              |         |")
    print("  |  1x1       1x1->3x3      1x1->5x5     MaxPool->1x1    |")
    print("  |    |         |              |              |         |")
    print("  |           [Concatenate output channels]             |")
    print("  +-----------------------------------------------------+")
    print()
    print("  1×1 convolutions (bottleneck): reduce channels cheaply")
    print("  Auxiliary classifiers at 2 points: vanishing gradient fix")
    print()
    versions = [
        ("Inception v1", "GoogLeNet (2014)",     "Original inception module"),
        ("Inception v2", "BN-Inception (2015)",  "Batch normalization"),
        ("Inception v3", "(2015)",               "Factorize 5x5->3x3+3x3; label smoothing"),
        ("Inception v4", "(2017)",               "Residual connections in inception"),
        ("Xception",     "(Chollet 2017)",       "Depthwise separable conv throughout"),
    ]
    for name, year, note in versions:
        print(f"  {name:<16} {year:<18} {note}")


# -- 5. ResNet (He 2015) -------------------------------------------------------
def resnet():
    print("\n=== ResNet (He et al., 2015) ===")
    print("  Problem: deeper networks had higher training error than shallower (degradation)")
    print("  Solution: skip connections / residual learning")
    print()
    print("  H(x) = F(x) + x")
    print("  Network learns residual F(x) = H(x) - x")
    print("  If identity is optimal, F->0 is easier than learning identity directly")
    print()
    print("  Residual Block:")
    print("    x ---> [Conv BN ReLU -> Conv BN] ---> (+) -> ReLU")
    print("    |                                    ^")
    print("    +------------------------------------+ (shortcut)")
    print()
    print("  Bottleneck Block (ResNet-50+): 1x1 -> 3x3 -> 1x1")

    variants = [
        ("ResNet-18",  18,  "2×(2+2+2+2) basic blocks",    11.7),
        ("ResNet-34",  34,  "2×(3+4+6+3) basic blocks",    21.8),
        ("ResNet-50",  50,  "3+4+6+3 bottleneck blocks",   25.6),
        ("ResNet-101", 101, "3+4+23+3 bottleneck blocks",  44.5),
        ("ResNet-152", 152, "3+8+36+3 bottleneck blocks",  60.2),
    ]
    print(f"\n  {'Model':<14} {'Layers':<8} {'Description':<35} {'Params(M)'}")
    for name, layers, desc, params in variants:
        print(f"  {name:<14} {layers:<8} {desc:<35} {params}")


# -- 6. DenseNet (Huang 2017) --------------------------------------------------
def densenet():
    print("\n=== DenseNet (Huang et al., 2017) ===")
    print("  Each layer receives feature maps from ALL preceding layers")
    print("  x_l = H_l([x_0, x_1, ..., x_{l-1}])")
    print()
    print("  Dense Block:")
    print("    x0 ---> [BN ReLU Conv] -> x1")
    print("    x0,x1 ---> [BN ReLU Conv] -> x2")
    print("    x0,x1,x2 ---> [BN ReLU Conv] -> x3")
    print()
    print("  Growth rate k: channels added per layer (typically k=12 or 32)")
    print("  Transition layers: 1×1 conv + avg pool between dense blocks")
    print()
    print("  Benefits:")
    print("    • Strong gradient flow through direct connections")
    print("    • Feature reuse: parameter efficient")
    print("    • Implicit deep supervision")
    print()
    variants = [
        ("DenseNet-121", 121, "6-12-24-16 dense layers",  8.0),
        ("DenseNet-169", 169, "6-12-32-32 dense layers", 14.1),
        ("DenseNet-201", 201, "6-12-48-32 dense layers", 20.0),
    ]
    print(f"  {'Model':<16} {'Layers':<8} {'Dense layers':<30} {'Params(M)'}")
    for name, layers, desc, params in variants:
        print(f"  {name:<16} {layers:<8} {desc:<30} {params}")


# -- 7. MobileNet / EfficientNet -----------------------------------------------
def efficient_nets():
    print("\n=== MobileNet & EfficientNet (Efficient Architectures) ===")
    print()
    print("  MobileNet v1 (Howard 2017):")
    print("    Depthwise separable convolutions; 28× fewer multiplications")
    print("    Width multiplier alpha, resolution multiplier rho for scaling")
    print()
    print("  MobileNet v2 (Sandler 2018):")
    print("    Inverted residuals: expand channels (6x) -> depthwise -> project")
    print("    Linear bottleneck: no ReLU after final 1×1 (preserves information)")
    print()
    print("  EfficientNet (Tan & Le 2019):")
    print("    Compound scaling: jointly scale depth, width, resolution")
    print("    phi controls total compute: depth=alphaphi, width=betaphi, resolution=gammaphi")
    print()
    variants = [
        ("EfficientNet-B0", 5.3,  "224×224",  77.1, "Baseline"),
        ("EfficientNet-B4", 19.3, "380×380",  82.9, "Recommended"),
        ("EfficientNet-B7", 66.3, "600×600",  84.3, "Best accuracy"),
    ]
    print(f"  {'Model':<20} {'Params(M)':<12} {'Input':<12} {'Top-1(%)':<10} {'Notes'}")
    for name, params, inp, acc, note in variants:
        print(f"  {name:<20} {params:<12} {inp:<12} {acc:<10} {note}")


# -- 8. Architecture timeline --------------------------------------------------
def timeline():
    print("\n=== CNN Architecture Timeline ===")
    arch = [
        (1998, "LeNet-5",        "~60K",   "~99.0 MNIST",       "First CNN"),
        (2012, "AlexNet",        "~62M",   "84.7 ImageNet",     "Deep learning era begins"),
        (2014, "VGG16",          "~138M",  "90.1",              "Depth > width"),
        (2014, "GoogLeNet",      "~5M",    "89.9",              "Inception module"),
        (2015, "ResNet-50",      "~25M",   "92.1",              "Skip connections"),
        (2017, "DenseNet-121",   "~8M",    "92.2",              "Dense connectivity"),
        (2017, "MobileNet v1",   "~4.2M",  "89.5",              "Mobile/embedded"),
        (2019, "EfficientNet-B7","~66M",   "97.1",              "Compound scaling"),
        (2020, "ViT-L/16",       "~307M",  "87.8",              "Pure transformer"),
        (2021, "ConvNeXt-L",     "~197M",  "87.5",              "CNN meets ViT design"),
    ]
    print(f"  {'Year':<6} {'Model':<20} {'Params':<10} {'Top-1/Acc':<16} {'Key idea'}")
    for year, name, params, acc, key in arch:
        print(f"  {year:<6} {name:<20} {params:<10} {acc:<16} {key}")


if __name__ == "__main__":
    lenet5()
    alexnet()
    vggnet()
    googlenet_inception()
    resnet()
    densenet()
    efficient_nets()
    timeline()
