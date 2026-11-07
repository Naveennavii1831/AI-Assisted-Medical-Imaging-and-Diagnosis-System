
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
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
MODEL_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"

DOCS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
BATCH_SIZE = 16

# Thresholds selected using the validation dataset.
# Do not tune these thresholds using the test dataset.
CONFIGURATIONS = [
    ("V2", "densenet121_chestmnist_v2.pth", 0.15),
    ("V3", "densenet121_chestmnist_v3.pth", 0.60),
]

torch.manual_seed(42)
np.random.seed(42)

print("=" * 65)
print("FINAL CHESTMNIST TEST-SET EVALUATION")
print("=" * 65)
print("Device:", DEVICE)

# --------------------------------------------------
# 2. Preprocessing
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
# 3. Load official ChestMNIST test dataset
# --------------------------------------------------

print("\nLoading official ChestMNIST test split...")

test_dataset = ChestMNIST(
    split="test",
    root=str(DATA_DIR),
    download=False,
    size=28,
    transform=transform,
)

if len(test_dataset) == 0:
    raise RuntimeError("The ChestMNIST test dataset is empty.")

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print("Test samples:", len(test_dataset))

# --------------------------------------------------
# 4. Evaluation function
# --------------------------------------------------

def evaluate_model(model, threshold):
    all_probabilities = []
    all_labels = []

    print("\nRunning inference...")

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(
            test_loader, start=1
        ):
            images = images.to(DEVICE)

            logits = model(images)
            probabilities = torch.sigmoid(logits)

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_labels.append(np.asarray(labels))

            if batch_index % 200 == 0:
                print(f"Processed {batch_index} batches")

    probabilities = np.concatenate(
        all_probabilities, axis=0
    )

    labels = np.concatenate(
        all_labels, axis=0
    ).astype(np.int32)

    predictions = (
        probabilities >= threshold
    ).astype(np.int32)

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

    results = {
        "threshold": threshold,
        "test_samples": int(len(labels)),
        "number_of_labels": int(labels.shape[1]),

        "element_accuracy": float(
            accuracy_score(
                labels.ravel(),
                predictions.ravel(),
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

        "macro_roc_auc": macro_auc,

        "binary_cross_entropy": float(
            nn.functional.binary_cross_entropy(
                torch.tensor(
                    probabilities,
                    dtype=torch.float32,
                ),
                torch.tensor(
                    labels,
                    dtype=torch.float32,
                ),
            ).item()
        ),

        "total_actual_positive_labels": int(labels.sum()),

        "total_predicted_positive_labels": int(
            predictions.sum()
        ),

        "clinical_validation": False,
    }

    return results


# --------------------------------------------------
# 5. Evaluate both models
# --------------------------------------------------

report = {
    "dataset": "ChestMNIST",
    "split": "official_test",
    "clinical_validation": False,
    "models": {},
}

for model_name, checkpoint_name, threshold in CONFIGURATIONS:
    model_path = MODEL_DIR / checkpoint_name

    print("\n" + "=" * 65)
    print(f"EVALUATING {model_name}")
    print("=" * 65)
    print("Checkpoint:", model_path.name)
    print("Threshold:", threshold)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model = models.densenet121(weights=None)

    model.classifier = nn.Linear(
        model.classifier.in_features,
        14,
    )

    # Training scripts saved checkpoints as dictionaries.
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    results = evaluate_model(model, threshold)

    report["models"][model_name] = {
        "checkpoint": checkpoint_name,
        **results,
    }

    for metric, value in results.items():
        print(f"{metric}: {value}")

    del model
    del checkpoint

# --------------------------------------------------
# 6. Save final report
# --------------------------------------------------

output_path = DOCS_DIR / "final_model_test_comparison.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)

print("\n" + "=" * 65)
print("FINAL TEST EVALUATION COMPLETED")
print("=" * 65)
print("Report saved to:", output_path)
print("Clinical validation: NOT performed")
