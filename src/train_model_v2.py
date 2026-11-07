
from pathlib import Path
import json
import time
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from torchvision.models import DenseNet121_Weights
from medmnist import ChestMNIST
from sklearn.metrics import precision_score, recall_score, f1_score


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")

BATCH_SIZE = 16
EPOCHS = 3
TRAIN_SAMPLES = 5000
VAL_SAMPLES = 1000
LEARNING_RATE = 0.001
THRESHOLD = 0.5

MODEL_PATH = MODEL_DIR / "densenet121_chestmnist_v2.pth"
HISTORY_PATH = DOCS_DIR / "training_history_v2.json"
RESULT_PATH = DOCS_DIR / "evaluation_v2.json"

torch.manual_seed(42)
np.random.seed(42)

print("=" * 65)
print("CHESTMNIST TRAINING V2")
print("=" * 65)
print("Device:", DEVICE)
print("Training samples:", TRAIN_SAMPLES)
print("Validation samples:", VAL_SAMPLES)
print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)


# --------------------------------------------------
# 2. Image preprocessing
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
# 3. Load training and validation datasets
# --------------------------------------------------

print("\nLoading datasets...")

train_dataset = ChestMNIST(
    split="train",
    root=str(DATA_DIR),
    download=False,
    size=28,
    transform=transform,
)

val_dataset = ChestMNIST(
    split="val",
    root=str(DATA_DIR),
    download=False,
    size=28,
    transform=transform,
)

train_count = min(TRAIN_SAMPLES, len(train_dataset))
val_count = min(VAL_SAMPLES, len(val_dataset))

# Deterministic subsets make this experiment reproducible.
train_subset = Subset(train_dataset, range(train_count))
val_subset = Subset(val_dataset, range(val_count))

train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print("Available training images:", len(train_dataset))
print("Available validation images:", len(val_dataset))
print("Images used for training:", train_count)
print("Images used for validation:", val_count)


# --------------------------------------------------
# 4. Initialize DenseNet121
# --------------------------------------------------

print("\nLoading pretrained DenseNet121...")

model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)

model.classifier = nn.Linear(
    model.classifier.in_features,
    14,
)

# Keep pretrained features frozen for this first comparison.
for parameter in model.features.parameters():
    parameter.requires_grad = False

model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE,
)


# --------------------------------------------------
# 5. Validation function
# --------------------------------------------------

def evaluate(model, loader):
    model.eval()

    total_loss = 0.0
    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.float().to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)

            all_probabilities.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    probabilities = np.concatenate(all_probabilities, axis=0)
    labels = np.concatenate(all_labels, axis=0).astype(np.int32)

    predictions = (probabilities >= THRESHOLD).astype(np.int32)

    metrics = {
        "validation_loss": float(total_loss / len(loader.dataset)),
        "element_accuracy": float(np.mean(predictions == labels)),
        "micro_precision": float(
            precision_score(
                labels, predictions, average="micro", zero_division=0
            )
        ),
        "micro_recall": float(
            recall_score(
                labels, predictions, average="micro", zero_division=0
            )
        ),
        "micro_f1": float(
            f1_score(
                labels, predictions, average="micro", zero_division=0
            )
        ),
    }

    return metrics


# --------------------------------------------------
# 6. Train and select the best validation checkpoint
# --------------------------------------------------

history = []
best_validation_loss = float("inf")
best_state = None

start_time = time.time()

print("\nStarting training...")

for epoch in range(EPOCHS):
    model.train()
    total_training_loss = 0.0

    for batch_index, (images, labels) in enumerate(train_loader, start=1):
        images = images.to(DEVICE)
        labels = labels.float().to(DEVICE)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_training_loss += loss.item() * images.size(0)

        if batch_index % 50 == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Batch {batch_index}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    average_training_loss = (
        total_training_loss / len(train_loader.dataset)
    )

    validation_metrics = evaluate(model, val_loader)

    epoch_record = {
        "epoch": epoch + 1,
        "training_loss": float(average_training_loss),
        **validation_metrics,
    }

    history.append(epoch_record)

    print(f"\nEpoch {epoch + 1}/{EPOCHS} complete")
    print(f"Training loss: {average_training_loss:.4f}")
    print(
        "Validation loss:",
        f"{validation_metrics['validation_loss']:.4f}",
    )
    print(
        "Validation micro F1:",
        f"{validation_metrics['micro_f1']:.4f}",
    )
    print(
        "Validation micro recall:",
        f"{validation_metrics['micro_recall']:.4f}",
    )

    if validation_metrics["validation_loss"] < best_validation_loss:
        best_validation_loss = validation_metrics["validation_loss"]
        best_state = copy.deepcopy(model.state_dict())

        torch.save(
            {
                "model_state_dict": best_state,
                "architecture": "densenet121",
                "num_labels": 14,
                "input_size": 64,
                "pretrained": True,
                "training_samples": train_count,
                "epochs_completed": epoch + 1,
                "best_validation_loss": best_validation_loss,
                "threshold": THRESHOLD,
            },
            MODEL_PATH,
        )

        print("Saved new best validation checkpoint.")

    print("-" * 65)


# --------------------------------------------------
# 7. Save training history and final validation metrics
# --------------------------------------------------

elapsed_seconds = round(time.time() - start_time, 2)

with open(HISTORY_PATH, "w", encoding="utf-8") as file:
    json.dump(history, file, indent=4)

best_model = models.densenet121(weights=None)
best_model.classifier = nn.Linear(
    best_model.classifier.in_features,
    14,
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)

best_model.load_state_dict(checkpoint["model_state_dict"])
best_model.to(DEVICE)

final_validation_metrics = evaluate(best_model, val_loader)

results = {
    "architecture": "DenseNet121",
    "dataset": "ChestMNIST",
    "training_samples": train_count,
    "validation_samples": val_count,
    "epochs_requested": EPOCHS,
    "epochs_completed": len(history),
    "best_validation_loss": best_validation_loss,
    "best_checkpoint": str(MODEL_PATH),
    "final_validation_metrics": final_validation_metrics,
    "training_history": history,
    "training_seconds": elapsed_seconds,
    "device": str(DEVICE),
    "threshold": THRESHOLD,
    "clinical_validation": False,
}

with open(RESULT_PATH, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("\n" + "=" * 65)
print("TRAINING COMPLETE")
print("=" * 65)
print("Best model saved to:", MODEL_PATH)
print("Training history saved to:", HISTORY_PATH)
print("Evaluation saved to:", RESULT_PATH)
print("Elapsed seconds:", elapsed_seconds)
print("Final validation metrics:", final_validation_metrics)
print("Clinical validation: NOT performed")
