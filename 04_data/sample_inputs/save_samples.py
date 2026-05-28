"""
Quick utility to save a few representative images for the demo/report.
Not essential for the main pipeline — just grabs some CIFAR-10 and PathMNIST
samples so we can show them in slides or the report without loading the full dataset.

Run from 03_code/:
    python ../04_data/sample_inputs/save_samples.py
"""
import os, sys
import torchvision
from PIL import Image

OUT = os.path.dirname(os.path.abspath(__file__))

cifar = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
classes = cifar.classes

for target_class in [0, 1, 3, 5]:
    count = 0
    for img, label in cifar:
        if label == target_class and count < 5:
            img.save(os.path.join(OUT, f"cifar10_{classes[target_class]}_{count}.png"))
            count += 1
        if count >= 5:
            break

try:
    import medmnist
    from medmnist import PathMNIST
    ds = PathMNIST(split="train", download=True, root="./data", as_rgb=True)
    pathmnist_classes = [
        "adipose", "background", "debris", "lymphocytes", "mucus",
        "smooth_muscle", "normal_mucosa", "stroma", "tumor"
    ]
    for target_class in [0, 1, 4]:
        count = 0
        for img, label in ds:
            lab = int(label.item()) if hasattr(label, "item") else int(label[0])
            if lab == target_class and count < 5:
                img.save(os.path.join(OUT, f"pathmnist_{pathmnist_classes[lab]}_{count}.png"))
                count += 1
            if count >= 5:
                break
except ImportError:
    print("medmnist not installed — skipping PathMNIST samples")

print(f"Saved sample images to {OUT}")
