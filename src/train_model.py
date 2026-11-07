from pathlib import Path
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from torchvision.models import DenseNet121_Weights
from medmnist import ChestMNIST


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")

BATCH_SIZE = 16
EPOCHS = 1
TRAIN_SAMPLES = 2000
VAL_SAMPLES = 500

torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("CHEST X-RAY CLASSIFICATION TRAINING")
print("=" * 60)
print("Device:", DEVICE)
print("Epochs:", EPOCHS)


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
    ),
])


# --------------------------------------------------
# 3. Load the official training and validation splits
# --------------------------------------------------

print("\nLoading dataset...")

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

print("Available training samples:", len(train_dataset))
print("Available validation samples:", len(val_dataset))

# Use deterministic subsets for the initial CPU prototype.
train_count = min(TRAIN_SAMPLES, len(train_dataset))
val_count = min(VAL_SAMPLES, len(val_dataset))

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

print("Training subset:", train_count)
print("Validation subset:", val_count)


# --------------------------------------------------
# 4. Load pretrained DenseNet121
# --------------------------------------------------

print("\nLoading pretrained DenseNet121...")
print("The first run may download pretrained weights.")

weights = DenseNet121_Weights.DEFAULT
model = models.densenet121(weights=weights)

# ChestMNIST contains 14 binary labels per image.
model.classifier = nn.Linear(
    model.classifier.in_features,
    14,
)

model = model.to(DEVICE)

# Freeze the pretrained feature extractor for the initial
# transfer-learning demonstration.
for parameter in model.features.parameters():
    parameter.requires_grad = False

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=0.001,
)


# --------------------------------------------------
# 5. Train the classification head
# --------------------------------------------------

print("\nStarting training...")
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for batch_index, (images, labels) in enumerate(train_loader):
        images = images.to(DEVICE)
        labels = labels.float().to(DEVICE)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_index + 1) % 25 == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Batch {batch_index + 1}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    average_loss = running_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1} complete | "
        f"Average training loss: {average_loss:.4f}"
    )


# --------------------------------------------------
# 6. Evaluate on the validation subset
# --------------------------------------------------

print("\nEvaluating on validation data...")

model.eval()

all_probabilities = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        logits = model(images)

        probabilities = torch.sigmoid(logits)

        all_probabilities.append(probabilities.cpu())
        all_labels.append(labels.float())

probabilities = torch.cat(all_probabilities).numpy()
labels = torch.cat(all_labels).numpy()

predictions = (probabilities >= 0.5).astype(np.int32)

element_accuracy = float(
    np.mean(predictions == labels)
)

average_validation_loss = float(
    nn.functional.binary_cross_entropy(
        torch.tensor(probabilities),
        torch.tensor(labels),
    ).item()
)

print("Validation element accuracy:", round(element_accuracy, 4))
print("Validation binary cross-entropy:", round(average_validation_loss, 4))


# --------------------------------------------------
# 7. Save the model and evaluation record
# --------------------------------------------------

model_path = MODEL_DIR / "densenet121_chestmnist.pth"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "architecture": "densenet121",
        "num_labels": 14,
        "input_size": 64,
        "pretrained": True,
    },
    model_path,
)

results = {
    "architecture": "DenseNet121",
    "pretrained": True,
    "dataset": "ChestMNIST",
    "training_samples": train_count,
    "validation_samples": val_count,
    "epochs": EPOCHS,
    "training_loss": float(average_loss),
    "validation_element_accuracy": element_accuracy,
    "validation_binary_cross_entropy": average_validation_loss,
    "threshold": 0.5,
    "device": str(DEVICE),
    "training_seconds": round(time.time() - start_time, 2),
    "clinical_validation": False,
}

results_path = DOCS_DIR / "initial_evaluation.json"

with open(results_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("\nTraining completed!")
print("Model saved to:", model_path)
print("Evaluation saved to:", results_path)
print("Clinical validation: NOT performed")