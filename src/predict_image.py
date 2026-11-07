
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import DenseNet121_Weights
from PIL import Image


# Project paths
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "densenet121_chestmnist.pth"

# CPU-compatible inference
DEVICE = torch.device("cpu")

# ChestMNIST label names
LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural Thickening",
    "Hernia",
]

# Match the preprocessing used during training
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model = models.densenet121(
        weights=None
    )

    model.classifier = nn.Linear(
        model.classifier.in_features,
        checkpoint["num_labels"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model


def predict(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    with Image.open(image_path) as source:
        image = source.convert("L")

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    model = load_model()

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)[0]

    print("\nAI-Assisted Medical Imaging System")
    print("=" * 45)
    print("Image:", image_path.name)
    print("Model: DenseNet121")
    print("Dataset: ChestMNIST")
    print("\nPredicted label probabilities:")

    results = sorted(
        zip(LABELS, probabilities.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )

    for label, probability in results:
        print(f"{label:22s}: {probability:.4f}")

    print("\nIMPORTANT:")
    print("Educational prototype only.")
    print("These predictions are NOT medical diagnoses.")
    print("Do not use this output for clinical decisions.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the ChestMNIST model on an image."
    )
    parser.add_argument(
        "image",
        help="Path to an image file, such as a PNG or JPEG.",
    )

    args = parser.parse_args()
    predict(args.image)
