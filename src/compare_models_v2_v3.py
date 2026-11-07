
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from medmnist import ChestMNIST
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"

OUTPUT_PATH = DOCS_DIR / "model_comparison_v2_v3.json"

DEVICE = torch.device("cpu")
BATCH_SIZE = 16

MODELS_TO_COMPARE = {
    "V2": {
        "path": MODEL_DIR / "densenet121_chestmnist_v2.pth",
        "threshold": 0.15,
    },
    "V3": {
        "path": MODEL_DIR / "densenet121_chestmnist_v3.pth",
        "threshold": 0.60,
    },
}


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
# 3. Load validation data
# --------------------------------------------------

print("=" * 65)
print("V2 VS V3 MODEL COMPARISON")
print("=" * 65)

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
# 4. Model evaluation
# --------------------------------------------------

def evaluate_model(model_path, threshold):
    model = models.densenet121(weights=None)

    model.classifier = nn.Linear(
        model.classifier.in_features,
        14,
    )

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)

            logits = model(images)

            probabilities = torch.sigmoid(logits)

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_labels.append(
                labels.numpy()
            )

    probabilities = np.concatenate(
        all_probabilities,
        axis=0,
    )

    labels = np.concatenate(
        all_labels,
        axis=0,
    ).astype(np.int32)

    predictions = (
        probabilities >= threshold
    ).astype(np.int32)

    metrics = {
        "threshold": threshold,
        "element_accuracy": float(
            np.mean(predictions == labels)
        ),
        "micro_precision": float(
            precision_score(
                labels.ravel(),
                predictions.ravel(),
                zero_division=0,
            )
        ),
        "micro_recall": float(
            recall_score(
                labels.ravel(),
                predictions.ravel(),
                zero_division=0,
            )
        ),
        "micro_f1": float(
            f1_score(
                labels.ravel(),
                predictions.ravel(),
                zero_division=0,
            )
        ),
        "macro_precision": float(
            precision_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "predicted_positive_labels": int(
            predictions.sum()
        ),
        "actual_positive_labels": int(
            labels.sum()
        ),
    }

    return metrics


# --------------------------------------------------
# 5. Compare models
# --------------------------------------------------

comparison = {
    "dataset": "ChestMNIST",
    "split": "validation",
    "validation_samples": len(dataset),
    "comparison": {},
    "clinical_validation": False,
}

for model_name, config in MODELS_TO_COMPARE.items():
    model_path = config["path"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {model_path}"
        )

    print(f"\nEvaluating {model_name}...")
    print("Checkpoint:", model_path.name)
    print("Threshold:", config["threshold"])

    metrics = evaluate_model(
        model_path,
        config["threshold"],
    )

    comparison["comparison"][model_name] = metrics

    print("Element accuracy:", f"{metrics['element_accuracy']:.4f}")
    print("Micro precision:", f"{metrics['micro_precision']:.4f}")
    print("Micro recall:", f"{metrics['micro_recall']:.4f}")
    print("Micro F1:", f"{metrics['micro_f1']:.4f}")
    print("Macro F1:", f"{metrics['macro_f1']:.4f}")
    print(
        "Predicted positive labels:",
        metrics["predicted_positive_labels"],
    )


# --------------------------------------------------
# 6. Save comparison report
# --------------------------------------------------

DOCS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        comparison,
        file,
        indent=4,
    )

print("\n" + "=" * 65)
print("MODEL COMPARISON COMPLETED")
print("=" * 65)
print("Report saved to:", OUTPUT_PATH)
print("Clinical validation: NOT performed")
