
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import DenseNet121_Weights

from medmnist import ChestMNIST
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "densenet121_chestmnist.pth"
DOCS_DIR = ROOT / "docs"

DOCS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
BATCH_SIZE = 16
THRESHOLD = 0.5

torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("CHESTMNIST TEST-SET EVALUATION")
print("=" * 60)

# --------------------------------------------------
# 2. Verify saved model
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Saved model not found: {MODEL_PATH}"
    )

print("Loading model:", MODEL_PATH)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)

# --------------------------------------------------
# 3. Preprocessing
# Must match training preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# --------------------------------------------------
# 4. Load official test split
# --------------------------------------------------

print("\nLoading ChestMNIST test split...")

test_dataset = ChestMNIST(
    split="test",
    root=str(DATA_DIR),
    download=False,
    size=28,
    transform=transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print("Test samples:", len(test_dataset))

if len(test_dataset) == 0:
    raise RuntimeError("The test dataset is empty.")

# --------------------------------------------------
# 5. Recreate model and load saved weights
# --------------------------------------------------

print("\nLoading DenseNet121 architecture...")

model = models.densenet121(weights=None)

model.classifier = nn.Linear(
    model.classifier.in_features,
    14,
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

# --------------------------------------------------
# 6. Generate predictions
# --------------------------------------------------

all_probabilities = []
all_labels = []

print("\nRunning inference...")

with torch.no_grad():
    for batch_index, (images, labels) in enumerate(test_loader):
        images = images.to(DEVICE)

        logits = model(images)
        probabilities = torch.sigmoid(logits)

        all_probabilities.append(probabilities.cpu().numpy())
        all_labels.append(np.asarray(labels))

        if (batch_index + 1) % 100 == 0:
            print(f"Processed {batch_index + 1} batches")

probabilities = np.concatenate(all_probabilities, axis=0)
labels = np.concatenate(all_labels, axis=0).astype(np.int32)

predictions = (
    probabilities >= THRESHOLD
).astype(np.int32)

print("Inference completed.")

# --------------------------------------------------
# 7. Calculate metrics
# --------------------------------------------------

# Element accuracy measures all label decisions together.
element_accuracy = float(
    accuracy_score(labels.ravel(), predictions.ravel())
)

# Macro metrics calculate each label's score independently
# and then give each label equal weight.
macro_precision = float(
    precision_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
)

macro_recall = float(
    recall_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
)

macro_f1 = float(
    f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
)

micro_precision = float(
    precision_score(
        labels.ravel(),
        predictions.ravel(),
        zero_division=0,
    )
)

micro_recall = float(
    recall_score(
        labels.ravel(),
        predictions.ravel(),
        zero_division=0,
    )
)

micro_f1 = float(
    f1_score(
        labels.ravel(),
        predictions.ravel(),
        zero_division=0,
    )
)

try:
    macro_auc = float(
        roc_auc_score(
            labels,
            probabilities,
            average="macro",
        )
    )
except ValueError:
    macro_auc = None

binary_cross_entropy = float(
    nn.functional.binary_cross_entropy(
        torch.tensor(probabilities, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
    ).item()
)

# --------------------------------------------------
# 8. Print results
# --------------------------------------------------

results = {
    "model": "DenseNet121",
    "dataset": "ChestMNIST",
    "split": "official_test",
    "test_samples": int(len(labels)),
    "number_of_labels": int(labels.shape[1]),
    "threshold": THRESHOLD,
    "element_accuracy": element_accuracy,
    "macro_precision": macro_precision,
    "macro_recall": macro_recall,
    "macro_f1": macro_f1,
    "micro_precision": micro_precision,
    "micro_recall": micro_recall,
    "micro_f1": micro_f1,
    "macro_roc_auc": macro_auc,
    "binary_cross_entropy": binary_cross_entropy,
    "clinical_validation": False,
}

print("\n" + "=" * 60)
print("TEST-SET RESULTS")
print("=" * 60)

for name, value in results.items():
    print(f"{name}: {value}")

# --------------------------------------------------
# 9. Save results
# --------------------------------------------------

output_path = DOCS_DIR / "test_evaluation.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("\nResults saved to:")
print(output_path)

print("\nClinical validation: NOT performed.")
