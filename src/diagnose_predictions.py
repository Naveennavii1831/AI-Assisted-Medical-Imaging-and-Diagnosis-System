
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import Compose, Grayscale, Normalize, Resize, ToTensor
from medmnist import ChestMNIST
from sklearn.metrics import precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "densenet121_chestmnist.pth"

BATCH_SIZE = 32
THRESHOLDS = [0.1, 0.2, 0.3, 0.5]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = Compose([
    Resize((64, 64)),
    Grayscale(num_output_channels=3),
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def main():
    print("=" * 65)
    print("CHESTMNIST PREDICTION DIAGNOSTICS")
    print("=" * 65)
    print(f"Device: {device}")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    dataset = ChestMNIST(
        root=str(ROOT / "data"),
        split="test",
        size=28,
        download=False,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = models.densenet121(weights=None)
    model.classifier = torch.nn.Linear(
        model.classifier.in_features, 14
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader, start=1):
            images = images.to(device)

            logits = model(images)
            probabilities = torch.sigmoid(logits)

            all_probabilities.append(probabilities.cpu().numpy())
            all_labels.append(np.asarray(labels))

            if batch_index % 100 == 0:
                print(f"Processed {batch_index} batches")

    probabilities = np.concatenate(all_probabilities, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    labels = (labels > 0.5).astype(np.int32)

    label_names = getattr(
        dataset,
        "labels",
        None,
    )

    # Print the positive-label distribution.
    print("\nTEST-DATASET LABEL DISTRIBUTION")
    print("-" * 65)

    for index in range(labels.shape[1]):
        actual_positive = int(labels[:, index].sum())
        actual_negative = int(len(labels) - actual_positive)

        print(
            f"Label {index + 1:02d}: "
            f"actual positives={actual_positive:6d}, "
            f"actual negatives={actual_negative:6d}"
        )

    print("\nPREDICTION THRESHOLD COMPARISON")
    print("-" * 65)

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(np.int32)

        predicted_positive = int(predictions.sum())
        actual_positive = int(labels.sum())

        precision = precision_score(
            labels, predictions, average="micro", zero_division=0
        )
        recall = recall_score(
            labels, predictions, average="micro", zero_division=0
        )
        f1 = f1_score(
            labels, predictions, average="micro", zero_division=0
        )

        print(f"\nThreshold: {threshold}")
        print(f"Predicted positive labels: {predicted_positive}")
        print(f"Actual positive labels:    {actual_positive}")
        print(f"Micro precision:           {precision:.4f}")
        print(f"Micro recall:              {recall:.4f}")
        print(f"Micro F1:                  {f1:.4f}")

    # Detailed per-label metrics at the original threshold.
    threshold = 0.5
    predictions = (probabilities >= threshold).astype(np.int32)

    print("\nPER-LABEL METRICS AT THRESHOLD 0.5")
    print("-" * 65)
    print("Label | Actual + | Predicted + | Precision | Recall | F1")

    for index in range(labels.shape[1]):
        actual_count = int(labels[:, index].sum())
        predicted_count = int(predictions[:, index].sum())

        precision = precision_score(
            labels[:, index],
            predictions[:, index],
            zero_division=0,
        )
        recall = recall_score(
            labels[:, index],
            predictions[:, index],
            zero_division=0,
        )
        f1 = f1_score(
            labels[:, index],
            predictions[:, index],
            zero_division=0,
        )

        print(
            f"{index + 1:5d} | "
            f"{actual_count:8d} | "
            f"{predicted_count:11d} | "
            f"{precision:9.4f} | "
            f"{recall:6.4f} | "
            f"{f1:.4f}"
        )

    print("\nDiagnostics completed.")
    print("These results are for educational model evaluation only.")


if __name__ == "__main__":
    main()
