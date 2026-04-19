"""
Working Example: CV Tools and Libraries
Covers OpenCV, Pillow, torchvision, albumentations, TIMM,
and major CV frameworks.
"""
import numpy as np
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_cv_tools")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_image(H=64, W=64, seed=0):
    rng = np.random.default_rng(seed)
    img = (rng.uniform(0, 1, (H, W, 3)) * 255).astype(np.uint8)
    img[10:30, 10:30, 0] = 220
    img[10:30, 10:30, 1:] = 50
    return img


# -- 1. OpenCV patterns --------------------------------------------------------
def opencv_patterns():
    print("=== OpenCV (cv2) Patterns ===")
    print("  Default channel order: BGR (not RGB!)")
    print()
    print("  Core operations:")
    code = [
        ("import cv2",                               "Import OpenCV"),
        ("img = cv2.imread('file.jpg')",             "Load as BGR uint8"),
        ("rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)","BGR -> RGB"),
        ("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)","BGR -> grayscale"),
        ("small = cv2.resize(img, (224, 224))",      "Resize to target"),
        ("blur = cv2.GaussianBlur(img, (5,5), 0)",   "Gaussian blur"),
        ("_, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)", "Thresholding"),
        ("edges = cv2.Canny(gray, 100, 200)",         "Canny edge detection"),
        ("cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)", "Draw bounding box"),
        ("cv2.imwrite('out.jpg', img)",               "Save image"),
        ("cap = cv2.VideoCapture(0)",                 "Open webcam"),
        ("ret, frame = cap.read()",                   "Read frame"),
    ]
    for c, d in code:
        print(f"  {c:<55} # {d}")

    print()
    print("  Try opencv-python demo (numpy fallback):")
    try:
        import cv2
        img = make_image()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(img, (32, 32))
        print(f"    cv2 available: {cv2.__version__}")
        print(f"    img: {img.shape}  gray: {gray.shape}  resized: {resized.shape}")
    except ImportError:
        img = make_image()
        # Grayscale with numpy
        gray = (0.114 * img[:,:,0] + 0.587 * img[:,:,1] + 0.299 * img[:,:,2]).astype(np.uint8)
        print(f"    cv2 not installed — numpy fallback")
        print(f"    img: {img.shape}  gray: {gray.shape}")


# -- 2. Pillow (PIL) patterns --------------------------------------------------
def pillow_patterns():
    print("\n=== Pillow (PIL) Patterns ===")
    print("  Default channel order: RGB")
    print()
    code = [
        ("from PIL import Image",                     "Import"),
        ("img = Image.open('file.jpg')",              "Open image"),
        ("img = img.convert('RGB')",                  "Ensure 3-channel RGB"),
        ("img = img.resize((224, 224))",              "Resize"),
        ("img = img.crop((x1, y1, x2, y2))",         "Crop"),
        ("img = img.rotate(angle=30)",                "Rotate"),
        ("img = img.transpose(Image.FLIP_LEFT_RIGHT)","Horizontal flip"),
        ("arr = np.array(img)",                       "PIL -> numpy (H, W, 3) uint8"),
        ("img = Image.fromarray(arr)",                "numpy -> PIL"),
        ("img.save('out.png')",                       "Save"),
    ]
    for c, d in code:
        print(f"  {c:<52} # {d}")

    print()
    try:
        from PIL import Image
        import io
        # Create in memory
        arr = make_image(32, 32)
        pil_img = Image.fromarray(arr)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        size = buf.tell()
        print(f"  PIL demo: img {pil_img.size}  mode={pil_img.mode}  PNG size={size} bytes")
    except ImportError:
        print("  Pillow not installed (pip install Pillow)")


# -- 3. torchvision transforms -------------------------------------------------
def torchvision_transforms():
    print("\n=== torchvision.transforms ===")
    print("  Standard training pipeline for PyTorch models:")
    print()
    code = """
    import torchvision.transforms as T

    train_transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.08, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),           # PIL/numpy -> float32 [0,1] (C, H, W)
        T.Normalize(mean=[0.485, 0.456, 0.406],    # ImageNet mean
                    std =[0.229, 0.224, 0.225]),    # ImageNet std
    ])

    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    """
    for line in code.strip().split("\n"):
        print(f"  {line}")
    print()
    print("  torchvision.transforms.v2 (new API, 2023):")
    print("    Supports tensors natively; composable with video/bbox/mask")

    try:
        import torchvision.transforms as T
        import torchvision
        print(f"\n  torchvision {torchvision.__version__} available")
        t = T.Compose([T.ToTensor()])
        arr = make_image(32, 32).astype(np.uint8)
        from PIL import Image
        pil = Image.fromarray(arr)
        tensor = t(pil)
        print(f"  ToTensor: PIL {arr.shape} -> tensor {tuple(tensor.shape)}  dtype={tensor.dtype}")
    except ImportError:
        print("  torchvision not installed (pip install torchvision)")


# -- 4. Albumentations --------------------------------------------------------
def albumentations_overview():
    print("\n=== Albumentations ===")
    print("  Fast, flexible, albumentations.ai — for segmentation/detection")
    print("  Applies same transform to image + mask + bounding boxes")
    print()
    code = """
    import albumentations as A

    transform = A.Compose([
        A.RandomResizedCrop(height=224, width=224, scale=(0.5, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.GridDistortion(p=0.1),          # elastic-like deformation
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.pytorch.ToTensorV2(),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

    # Apply
    transformed = transform(image=image, bboxes=bboxes, class_labels=labels)
    """
    for line in code.strip().split("\n"):
        print(f"  {line}")

    try:
        import albumentations as A
        arr = make_image(64, 64)
        t = A.Compose([A.HorizontalFlip(p=1.0), A.Resize(32, 32)])
        out = t(image=arr)["image"]
        print(f"\n  albumentations demo: {arr.shape} -> {out.shape}")
    except ImportError:
        print("\n  albumentations not installed (pip install albumentations)")


# -- 5. TIMM (PyTorch Image Models) -------------------------------------------
def timm_overview():
    print("\n=== TIMM (timm.fast.ai) ===")
    print("  Collection of 700+ pretrained image models")
    print()
    code = [
        ("import timm",                                       "Import"),
        ("timm.list_models('resnet*')",                      "List matching models"),
        ("model = timm.create_model('resnet50', pretrained=True)",  "Load pretrained"),
        ("model = timm.create_model('efficientnet_b3', num_classes=10)", "Custom classes"),
        ("model = timm.create_model('vit_base_patch16_224')",        "Vision Transformer"),
        ("cfg = model.default_cfg",                          "Get model config"),
        ("feats = model.forward_features(x)",                "Get feature maps"),
        ("model.reset_classifier(num_classes=5)",            "Swap head"),
    ]
    for c, d in code:
        print(f"  {c:<60} # {d}")
    print()
    print("  Popular TIMM models:")
    models = [
        ("resnet50",                  "77M, classic ResNet; baseline for comparison"),
        ("efficientnet_b4",           "19M, efficient scaling; good acc/speed"),
        ("vit_base_patch16_224",      "86M, ViT; strong with large data"),
        ("swin_base_patch4_window7_224","88M, Swin Transformer; local attention"),
        ("convnext_base",             "89M, ConvNet redesigned as ViT"),
        ("deit3_base_patch16_224",    "87M, Data-efficient ViT; strong baselines"),
        ("eva02_base_patch14_448",    "86M, EVA-02; sota transfer learning"),
    ]
    for m, d in models:
        print(f"    {m:<35} {d}")

    try:
        import timm
        print(f"\n  TIMM {timm.__version__} installed")
        n = len(timm.list_models())
        print(f"  Available models: {n}")
    except ImportError:
        print("\n  TIMM not installed (pip install timm)")


# -- 6. Detection / segmentation frameworks -----------------------------------
def detection_frameworks():
    print("\n=== Detection & Segmentation Frameworks ===")
    frameworks = [
        ("Ultralytics (YOLOv8/11)",
         "pip install ultralytics",
         "from ultralytics import YOLO\n"
         "    model = YOLO('yolov8n.pt')  # detection\n"
         "    results = model.predict('image.jpg')"),
        ("Detectron2 (Meta)",
         "pip install detectron2",
         "from detectron2 import model_zoo\n"
         "    cfg = model_zoo.get_config('COCO-Detection/faster_rcnn_R_50_FPN.yaml')\n"
         "    predictor = DefaultPredictor(cfg)"),
        ("MMDetection (OpenMMLab)",
         "pip install mmdet",
         "from mmdet.apis import init_detector, inference_detector"),
        ("HuggingFace Transformers (detection)",
         "pip install transformers",
         "from transformers import pipeline\n"
         "    od = pipeline('object-detection', model='facebook/detr-resnet-50')"),
    ]
    for name, install, example in frameworks:
        print(f"\n  {name}")
        print(f"    Install: {install}")
        print(f"    Usage:")
        for line in example.split("\n"):
            print(f"      {line}")


if __name__ == "__main__":
    opencv_patterns()
    pillow_patterns()
    torchvision_transforms()
    albumentations_overview()
    timm_overview()
    detection_frameworks()
