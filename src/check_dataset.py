from pathlib import Path

import medmnist
import numpy as np
from medmnist import INFO
from medmnist import ChestMNIST


# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Select the dataset
DATA_FLAG = "chestmnist"
info = INFO[DATA_FLAG]

print("=" * 60)
print("AI-ASSISTED MEDICAL IMAGING SYSTEM")
print("=" * 60)

print("Dataset:", DATA_FLAG)
print("Task:", info["task"])
print("Number of classes:", len(info["label"]))

print("\nDownloading/loading training data...")

# Download the training split
train_dataset = ChestMNIST(
    split="train",
    root=str(DATA_DIR),
    download=True,
    size=28,
)

print("\nDataset loaded successfully!")

print("Training images:", len(train_dataset))
print("Image shape:", train_dataset.imgs.shape)
print("Label shape:", train_dataset.labels.shape)

# Inspect one example
image, label = train_dataset[0]

image_array = np.asarray(image)
label_array = np.asarray(label)

print("\nFirst sample")
print("Image dimensions:", image_array.shape)
print("Image data type:", image_array.dtype)
print("Label dimensions:", label_array.shape)
print("Label values:", label_array)

print("\nDataset verification completed.")