
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import DenseNet121_Weights
from sklearn.metrics import precision_score, recall_score, f1_score
from medmnist import ChestMNIST


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "densenet121_chestmnist_v2.pth"
OUTPUT_PATH = ROOT / "docs" / "threshold_tuning_v2.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16

THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


# --------------------------------------------------
# 2. Image preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# 3. Load the validation dataset
# --------------------------------------------------

print("=" * 65)
print("DENSENET121 V2 VALIDATION THRESHOLD TUNING")
print("=" * 65)

print(f"Device: {DEVICE}")
print(f"Model: {MODEL_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL_PATH}"
    )

dataset = ChestMNIST(
    split="val",
    root=str(DATA_DIR),
    download=False,
    size=28,
    transform=transform
)

print(f"Validation samples: {len(dataset)}")

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# 4. Rebuild the model architecture and load V2
# --------------------------------------------------

print("\nLoading DenseNet121...")

model = models.densenet121(weights=None)
model.classifier = nn.Linear(
    model.classifier.in_features,
    14
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()


# --------------------------------------------------
# 5. Collect probabilities and actual labels
# --------------------------------------------------

all_probabilities = []
all_labels = []

print("\nRunning inference on validation data...")

with torch.no_grad():
    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(DEVICE)

        logits = model(images)
        probabilities = torch.sigmoid(logits)

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

        all_labels.append(
            labels.cpu().numpy()
        )

        if batch_index % 10 == 0:
            print(f"Processed {batch_index} batches")


probabilities = np.concatenate(all_probabilities, axis=0)
labels = np.concatenate(all_labels, axis=0)

labels = labels.astype(np.int32)

print("\nInference completed.")
print(f"Probability array shape: {probabilities.shape}")
print(f"Label array shape: {labels.shape}")


# --------------------------------------------------
# 6. Evaluate each threshold
# --------------------------------------------------

results = {}

print("\nTHRESHOLD COMPARISON")
print("-" * 90)

print(
    f"{'Threshold':<12}"
    f"{'Precision':<15}"
    f"{'Recall':<15}"
    f"{'F1-score':<15}"
    f"{'Predicted +':<15}"
)

best_threshold = None
best_f1 = -1.0

for threshold in THRESHOLDS:

    predictions = (
        probabilities >= threshold
    ).astype(np.int32)

    precision = precision_score(
        labels.ravel(),
        predictions.ravel(),
        zero_division=0
    )

    recall = recall_score(
        labels.ravel(),
        predictions.ravel(),
        zero_division=0
    )

    f1 = f1_score(
        labels.ravel(),
        predictions.ravel(),
        zero_division=0
    )

    predicted_positive = int(predictions.sum())

    results[str(threshold)] = {
        "micro_precision": float(precision),
        "micro_recall": float(recall),
        "micro_f1": float(f1),
        "predicted_positive_labels": predicted_positive
    }

    print(
        f"{threshold:<12.2f}"
        f"{precision:<15.4f}"
        f"{recall:<15.4f}"
        f"{f1:<15.4f}"
        f"{predicted_positive:<15}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold


# --------------------------------------------------
# 7. Save the results
# --------------------------------------------------

output = {
    "model": "DenseNet121",
    "checkpoint": MODEL_PATH.name,
    "dataset": "ChestMNIST",
    "split": "validation",
    "validation_samples": int(len(dataset)),
    "number_of_labels": 14,
    "selection_metric": "micro_f1",
    "thresholds_tested": THRESHOLDS,
    "best_global_threshold_by_micro_f1": float(best_threshold),
    "best_validation_micro_f1": float(best_f1),
    "threshold_results": results,
    "clinical_validation": False
}

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(output, file, indent=4)


# --------------------------------------------------
# 8. Final summary
# --------------------------------------------------

print("\n" + "=" * 65)
print("THRESHOLD TUNING COMPLETED")
print("=" * 65)

print(f"Best global threshold by validation micro F1: {best_threshold:.2f}")
print(f"Best validation micro F1: {best_f1:.4f}")
print(f"Results saved to: {OUTPUT_PATH}")

print("\nIMPORTANT:")
print("The selected threshold is based on validation data only.")
print("It is not clinically validated.")
print("Do not use this educational model for medical decisions.")
