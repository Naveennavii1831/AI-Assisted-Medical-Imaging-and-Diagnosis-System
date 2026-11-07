
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from medmnist import ChestMNIST
from sklearn.metrics import precision_score, recall_score, f1_score


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "densenet121_chestmnist_v3.pth"
OUTPUT_PATH = ROOT / "docs" / "threshold_tuning_v3.json"

DEVICE = torch.device("cpu")
BATCH_SIZE = 16

THRESHOLDS = [
    0.15, 0.20, 0.25, 0.30, 0.35,
    0.40, 0.45, 0.50, 0.60, 0.70, 0.80
]

print("=" * 65)
print("DENSENET121 V3 VALIDATION THRESHOLD TUNING")
print("=" * 65)


# --------------------------------------------------
# 2. Preprocessing
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
# 3. Load validation dataset
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL_PATH}"
    )

dataset = ChestMNIST(
    split="val",
    root=str(DATA_DIR),
    download=False,
    size=28,
    transform=transform,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print("Device:", DEVICE)
print("Validation samples:", len(dataset))


# --------------------------------------------------
# 4. Load V3 model
# --------------------------------------------------

model = models.densenet121(weights=None)

model.classifier = nn.Linear(
    model.classifier.in_features,
    14,
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()


# --------------------------------------------------
# 5. Generate validation probabilities
# --------------------------------------------------

all_probabilities = []
all_labels = []

print("\nRunning inference...")

with torch.no_grad():
    for batch_index, (images, labels) in enumerate(
        loader,
        start=1,
    ):
        images = images.to(DEVICE)

        logits = model(images)

        probabilities = torch.sigmoid(logits)

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

        all_labels.append(
            labels.numpy()
        )

        if batch_index % 100 == 0:
            print(f"Processed {batch_index} batches")

probabilities = np.concatenate(
    all_probabilities,
    axis=0,
)

labels = np.concatenate(
    all_labels,
    axis=0,
).astype(np.int32)

print("Inference completed.")
print("Probability shape:", probabilities.shape)


# --------------------------------------------------
# 6. Evaluate thresholds
# --------------------------------------------------

results = {}

best_threshold = None
best_f1 = -1.0

print("\nTHRESHOLD COMPARISON")
print("-" * 80)

print(
    f"{'Threshold':<12}"
    f"{'Precision':<15}"
    f"{'Recall':<15}"
    f"{'Micro F1':<15}"
    f"{'Predicted +':<15}"
)

for threshold in THRESHOLDS:

    predictions = (
        probabilities >= threshold
    ).astype(np.int32)

    precision = precision_score(
        labels.ravel(),
        predictions.ravel(),
        zero_division=0,
    )

    recall = recall_score(
        labels.ravel(),
        predictions.ravel(),
        zero_division=0,
    )

    f1 = f1_score(
        labels.ravel(),
        predictions.ravel(),
        zero_division=0,
    )

    predicted_positive = int(
        predictions.sum()
    )

    results[str(threshold)] = {
        "micro_precision": float(precision),
        "micro_recall": float(recall),
        "micro_f1": float(f1),
        "predicted_positive_labels": predicted_positive,
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
# 7. Save results
# --------------------------------------------------

output = {
    "model": "DenseNet121",
    "checkpoint": MODEL_PATH.name,
    "dataset": "ChestMNIST",
    "split": "validation",
    "validation_samples": len(dataset),
    "thresholds_tested": THRESHOLDS,
    "selection_metric": "micro_f1",
    "best_global_threshold_by_micro_f1": best_threshold,
    "best_validation_micro_f1": float(best_f1),
    "threshold_results": results,
    "clinical_validation": False,
}

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        output,
        file,
        indent=4,
    )

print("\n" + "=" * 65)
print("V3 THRESHOLD TUNING COMPLETED")
print("=" * 65)
print("Best threshold:", best_threshold)
print("Best validation micro F1:", round(best_f1, 4))
print("Results saved to:", OUTPUT_PATH)
print("Clinical validation: NOT performed")
