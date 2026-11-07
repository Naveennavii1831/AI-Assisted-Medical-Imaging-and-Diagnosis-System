
from pathlib import Path
import base64
import io
import sqlite3
import uuid

import numpy as np
import pydicom
import torch
import torch.nn as nn

from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

# Correct import when starting with:
# python -m uvicorn src.api:app --reload
from src.review_database import (
    initialize_database,
    save_review,
    get_reviews,
)


# =========================================================
# 1. CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "densenet121_chestmnist.pth"

DEVICE = torch.device("cpu")

MAX_FILE_SIZE = 10 * 1024 * 1024

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

TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# =========================================================
# 2. FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="AI-Assisted Medical Imaging API",
    description=(
        "Educational ChestMNIST image-classification prototype "
        "with DICOM conversion, experimental Grad-CAM, and "
        "a persistent reviewer decision log. Not for clinical use."
    ),
    version="0.4.0",
)

model = None


# =========================================================
# 3. REVIEW REQUEST VALIDATION
# =========================================================

class ReviewRequest(BaseModel):
    review_id: str = Field(min_length=1, max_length=100)
    filename: str = Field(min_length=1, max_length=255)
    decision: str = Field(min_length=1, max_length=20)
    reviewer: str = Field(min_length=1, max_length=100)
    comments: str = Field(default="", max_length=2000)


# =========================================================
# 4. LOAD MODEL
# =========================================================

def load_model_instance():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    loaded_model = models.densenet121(weights=None)

    loaded_model.classifier = nn.Linear(
        loaded_model.classifier.in_features,
        checkpoint["num_labels"],
    )

    loaded_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    loaded_model.to(DEVICE)
    loaded_model.eval()

    return loaded_model


# =========================================================
# 5. APPLICATION STARTUP
# =========================================================

@app.on_event("startup")
def startup_application():
    global model

    initialize_database()
    model = load_model_instance()


# =========================================================
# 6. BASIC API ENDPOINTS
# =========================================================

@app.get("/")
def home():
    return {
        "message": "AI-Assisted Medical Imaging API is running",
        "status": "educational_prototype",
        "supported_formats": [
            "PNG",
            "JPEG",
            "WebP",
            "DICOM",
        ],
        "clinical_use": False,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "clinical_use": False,
    }


# =========================================================
# 7. DICOM CONVERSION
# =========================================================

def convert_dicom_to_image(contents):
    """
    Convert a compatible single-frame, two-dimensional
    DICOM image to grayscale and RGB PIL images.

    This is basic display normalization, not full
    clinical DICOM windowing.
    """

    try:
        dataset = pydicom.dcmread(
            io.BytesIO(contents),
            force=False,
        )

        if "PixelData" not in dataset:
            raise ValueError(
                "The DICOM file contains no image pixel data."
            )

        pixels = dataset.pixel_array

    except Exception as exc:
        raise ValueError(
            f"Unable to read DICOM pixel data: {exc}"
        ) from exc

    if pixels.ndim != 2:
        raise ValueError(
            "Only single-frame 2D DICOM images are supported. "
            f"Received pixel array shape: {pixels.shape}"
        )

    pixels = pixels.astype(np.float32)

    slope = float(getattr(dataset, "RescaleSlope", 1))
    intercept = float(getattr(dataset, "RescaleIntercept", 0))

    pixels = pixels * slope + intercept

    lower = float(np.min(pixels))
    upper = float(np.max(pixels))

    if upper > lower:
        pixels = (
            (pixels - lower)
            / (upper - lower)
            * 255.0
        )
    else:
        pixels = np.zeros_like(pixels)

    pixels = np.clip(
        pixels,
        0,
        255,
    ).astype(np.uint8)

    if (
        getattr(dataset, "PhotometricInterpretation", "")
        == "MONOCHROME1"
    ):
        pixels = 255 - pixels

    grayscale_image = Image.fromarray(pixels).convert("L")
    original_image = grayscale_image.convert("RGB")

    return grayscale_image, original_image


# =========================================================
# 8. GRAD-CAM
# =========================================================

def create_gradcam(image_tensor, class_index, original_image):
    """
    Generate an experimental Grad-CAM overlay.

    Returns a PNG data URL.
    """

    if model is None:
        raise RuntimeError("The model is not loaded.")

    activations = {}
    gradients = {}

    def save_activation(_module, _inputs, output):
        activations["value"] = output

    def save_gradient(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    target_layer = model.features.denseblock4

    activation_hook = target_layer.register_forward_hook(
        save_activation
    )

    gradient_hook = target_layer.register_full_backward_hook(
        save_gradient
    )

    try:
        model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            logits = model(image_tensor)
            logits[0, class_index].backward()

        feature_maps = activations["value"][0]
        grads = gradients["value"][0]

        weights = grads.mean(dim=(1, 2))

        heatmap = torch.relu(
            (
                weights[:, None, None] * feature_maps
            ).sum(dim=0)
        )

        heatmap = heatmap.detach().cpu().numpy()
        heatmap -= heatmap.min()

        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        heatmap_image = Image.fromarray(
            (heatmap * 255).astype(np.uint8)
        ).resize(
            original_image.size,
            Image.Resampling.BILINEAR,
        )

        heatmap_array = (
            np.asarray(heatmap_image).astype(np.float32)
            / 255.0
        )

        base = np.asarray(
            original_image.convert("RGB")
        ).astype(np.float32)

        color_map = np.zeros_like(base)
        color_map[..., 0] = 255
        color_map[..., 1] = (1 - heatmap_array) * 180
        color_map[..., 2] = (1 - heatmap_array) * 40

        overlay = (
            0.60 * base
            + 0.40 * color_map * heatmap_array[..., None]
        )

        overlay = np.clip(
            overlay,
            0,
            255,
        ).astype(np.uint8)

        output = io.BytesIO()

        Image.fromarray(overlay).save(
            output,
            format="PNG",
        )

        encoded = base64.b64encode(
            output.getvalue()
        ).decode("ascii")

        return f"data:image/png;base64,{encoded}"

    finally:
        activation_hook.remove()
        gradient_hook.remove()
        model.zero_grad(set_to_none=True)


# =========================================================
# 9. IMAGE PREDICTION
# =========================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    filename = Path(
        file.filename or "uploaded_image"
    ).name

    extension = Path(filename).suffix.lower()

    contents = await file.read(MAX_FILE_SIZE + 1)

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File must be 10 MB or smaller.",
        )

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    is_dicom = (
        extension == ".dcm"
        or file.content_type in {
            "application/dicom",
            "application/dicom+json",
        }
    )

    try:
        if is_dicom:
            image, original_image = convert_dicom_to_image(
                contents
            )
            source_format = "DICOM"

        else:
            if file.content_type not in {
                "image/png",
                "image/jpeg",
                "image/webp",
            }:
                raise HTTPException(
                    status_code=415,
                    detail=(
                        "Unsupported file type. Upload PNG, "
                        "JPEG, WebP, or a .dcm DICOM file."
                    ),
                )

            with Image.open(io.BytesIO(contents)) as source:
                source.load()
                original_image = source.convert("RGB")
                image = source.convert("L")

            source_format = "standard_image"

    except HTTPException:
        raise

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to process the uploaded image: {exc}",
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Image processing failed: {exc}",
        ) from exc

    image_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)[0].tolist()

    predictions = [
        {
            "label": label,
            "probability": round(float(probability), 4),
        }
        for label, probability in zip(LABELS, probabilities)
    ]

    predictions.sort(
        key=lambda item: item["probability"],
        reverse=True,
    )

    target_index = int(np.argmax(probabilities))

    try:
        heatmap = create_gradcam(
            image_tensor,
            target_index,
            original_image,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Grad-CAM generation failed: {exc}",
        ) from exc

    review_id = str(uuid.uuid4())

    return {
        "review_id": review_id,
        "filename": filename,
        "source_format": source_format,
        "model": "DenseNet121",
        "dataset": "ChestMNIST",
        "predictions": predictions,
        "gradcam_image": heatmap,
        "gradcam_label": LABELS[target_index],
        "notice": (
            "Educational prototype only. This ChestMNIST-trained "
            "model is not clinically validated. Predictions must "
            "not be used for clinical decisions. DICOM support is "
            "limited to compatible single-frame 2D images and does "
            "not provide full clinical windowing or CT/MRI series "
            "processing."
        ),
    }


# =========================================================
# 10. SAVE REVIEW DECISION
# =========================================================

@app.post("/reviews", status_code=201)
def create_review(review: ReviewRequest):
    decision = review.decision.strip().lower()
    reviewer = review.reviewer.strip()
    filename = review.filename.strip()
    review_id = review.review_id.strip()
    comments = review.comments.strip()

    if decision not in {"approved", "rejected"}:
        raise HTTPException(
            status_code=422,
            detail="Decision must be 'approved' or 'rejected'.",
        )

    if not review_id or not filename or not reviewer:
        raise HTTPException(
            status_code=422,
            detail="Review ID, filename, and reviewer are required.",
        )

    try:
        saved_review = save_review(
            review_id=review_id,
            filename=filename,
            decision=decision,
            reviewer=reviewer,
            comments=comments,
        )

    except sqlite3.IntegrityError as exc:
        raise HTTPException(
            status_code=409,
            detail="A review already exists for this review ID.",
        ) from exc

    return {
        "message": "Review saved successfully.",
        "review": saved_review,
    }


# =========================================================
# 11. REVIEW HISTORY
# =========================================================

@app.get("/reviews")
def list_reviews(limit: int = 50):
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=422,
            detail="Limit must be between 1 and 100.",
        )

    reviews = get_reviews(limit=limit)

    return {
        "count": len(reviews),
        "reviews": reviews,
    }


# =========================================================
# 12. RETRIEVE ONE REVIEW
# =========================================================

@app.get("/reviews/{review_id}")
def retrieve_review(review_id: str):
    reviews = get_reviews(limit=100)

    for review in reviews:
        if review["review_id"] == review_id:
            return review

    raise HTTPException(
        status_code=404,
        detail="Review not found.",
    )
