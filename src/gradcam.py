from pathlib import Path
import argparse

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "densenet121_chestmnist.pth"
OUTPUT_PATH = ROOT / "data" / "sample_images" / "gradcam_heatmap.png"

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

DEVICE = torch.device("cpu")

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
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(
        model.classifier.in_features,
        checkpoint["num_labels"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model


def generate_gradcam(image_path, target_label=None):
    model = load_model()
    activations = {}
    gradients = {}

    def save_activation(_module, _inputs, output):
        activations["value"] = output

    def save_gradient(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    target_layer = model.features.denseblock4
    activation_hook = target_layer.register_forward_hook(save_activation)
    gradient_hook = target_layer.register_full_backward_hook(save_gradient)

    try:
        with Image.open(image_path) as source:
            original = source.convert("RGB")
            input_image = transform(source.convert("L")).unsqueeze(0).to(DEVICE)

        model.zero_grad(set_to_none=True)

        logits = model(input_image)
        probabilities = torch.sigmoid(logits)[0]

        if target_label is None:
            class_index = int(torch.argmax(probabilities).item())
        else:
            if target_label not in LABELS:
                raise ValueError(
                    f"Unknown label '{target_label}'. Choose one of: {LABELS}"
                )
            class_index = LABELS.index(target_label)

        logits[0, class_index].backward()

        feature_maps = activations["value"][0]
        grads = gradients["value"][0]

        weights = grads.mean(dim=(1, 2))
        heatmap = torch.relu(
            (weights[:, None, None] * feature_maps).sum(dim=0)
        )

        heatmap = heatmap.detach().cpu().numpy()
        heatmap -= heatmap.min()

        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        heatmap_image = Image.fromarray(
            np.uint8(heatmap * 255)
        ).resize(original.size, Image.Resampling.BILINEAR)

        heatmap_array = np.asarray(heatmap_image).astype(np.float32) / 255.0

        # Create a simple red/orange heatmap overlay without extra libraries.
        red = np.full_like(heatmap_array, 255)
        green = (1.0 - heatmap_array) * 180
        blue = (1.0 - heatmap_array) * 40
        color_map = np.stack([red, green, blue], axis=-1)

        base = np.asarray(original).astype(np.float32)
        overlay = (
            0.55 * base + 0.45 * color_map * heatmap_array[..., None]
        )
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(OUTPUT_PATH)

        print(f"Selected output: {LABELS[class_index]}")
        print(f"Model output value: {probabilities[class_index].item():.4f}")
        print(f"Grad-CAM image saved to: {OUTPUT_PATH}")
        print(
            "Note: This visualization is experimental and does not establish "
            "medical evidence or diagnostic accuracy."
        )

    finally:
        activation_hook.remove()
        gradient_hook.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an experimental Grad-CAM visualization."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=str(ROOT / "data" / "sample_images" / "chestmnist_sample.png"),
        help="Path to the input image.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional output label, such as Infiltration or Effusion.",
    )
    args = parser.parse_args()

    generate_gradcam(args.image, args.label)