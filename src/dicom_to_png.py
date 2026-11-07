
from pathlib import Path
import argparse

import numpy as np
import pydicom
from PIL import Image


def convert_dicom_to_png(dicom_path, output_path):
    dicom_path = Path(dicom_path)
    output_path = Path(output_path)

    if not dicom_path.is_file():
        raise FileNotFoundError(f"DICOM file not found: {dicom_path}")

    # Read the DICOM file.
    dataset = pydicom.dcmread(dicom_path)

    if not hasattr(dataset, "PixelData"):
        raise ValueError("This DICOM file does not contain pixel data.")

    # This initial version supports single-frame 2D images only.
    pixels = dataset.pixel_array

    if pixels.ndim != 2:
        raise ValueError(
            f"Expected a single 2D image, but received shape {pixels.shape}. "
            "Multi-frame images and image series need additional processing."
        )

    # Convert stored pixel values into floating-point values.
    pixels = pixels.astype(np.float32)

    # Apply rescale parameters when provided by the DICOM file.
    slope = float(getattr(dataset, "RescaleSlope", 1))
    intercept = float(getattr(dataset, "RescaleIntercept", 0))
    pixels = pixels * slope + intercept

    # Normalize the image into the 0–255 range for PNG display.
    lower = float(np.min(pixels))
    upper = float(np.max(pixels))

    if upper > lower:
        pixels = (pixels - lower) / (upper - lower) * 255
    else:
        pixels = np.zeros_like(pixels)

    pixels = np.clip(pixels, 0, 255).astype(np.uint8)

    # MONOCHROME1 images use the opposite display intensity convention.
    if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = 255 - pixels

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(output_path)

    print(f"Input DICOM: {dicom_path}")
    print(f"Image dimensions: {pixels.shape[1]} x {pixels.shape[0]}")
    print(f"PNG saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a single-frame 2D DICOM image to PNG."
    )
    parser.add_argument("dicom_file", help="Path to the input DICOM file")
    parser.add_argument(
        "--output",
        default="data/sample_images/dicom_converted.png",
        help="Output PNG path",
    )

    args = parser.parse_args()

    try:
        convert_dicom_to_png(args.dicom_file, args.output)
    except Exception as error:
        raise SystemExit(f"Conversion failed: {error}")
