from pathlib import Path

from medmnist import ChestMNIST
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = DATA_DIR / "sample_images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

dataset = ChestMNIST(
    split="test",
    root=str(DATA_DIR),
    download=True,
    size=28,
)

image, label = dataset[0]

output_path = OUTPUT_DIR / "chestmnist_sample.png"

image = image.convert("L")
image.save(output_path)

print("Sample image exported successfully!")
print("Image path:", output_path)
print("Image size:", image.size)
print("Dataset labels:", label)