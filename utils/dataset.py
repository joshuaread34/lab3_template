from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

IMG_SIZE = 227
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class PetNoseDataset(Dataset):
    def __init__(self, root="data/oxford-iiit-pet-noses", split="train"):
        super().__init__()

        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")

        root = Path(root)
        self.img_dir = root / "images-original" / "images"
        label_file = root / f"{split}_noses.txt"

        self.samples = []

        # Each line looks like:
        # shiba_inu_136.jpg,"(182, 203)"
        with open(label_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                filename, coordinates = line.split(",", 1)
                filename = filename.strip().strip('"')
                coordinates = coordinates.strip().strip('"')
                coordinates = coordinates.strip("()")

                u_str, v_str = coordinates.split(",", 1)
                u = float(u_str.strip())
                v = float(v_str.strip())

                self.samples.append((filename, u, v))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, u, v = self.samples[idx]

        image_path = self.img_dir / filename
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            original_width, original_height = img.size

            # Resize the image to 227 x 227.
            img = img.resize(
                (IMG_SIZE, IMG_SIZE),
                Image.BILINEAR
            )

        # Scale the nose coordinates to match the resized image.
        u_resized = u * IMG_SIZE / original_width
        v_resized = v * IMG_SIZE / original_height

        # Convert to a tensor and apply ImageNet normalization.
        image = TF.to_tensor(img)
        image = TF.normalize(image, mean=MEAN, std=STD)

        label = torch.tensor(
            [u_resized, v_resized],
            dtype=torch.float32
        )
        orig_size = torch.tensor(
            [original_width, original_height],
            dtype=torch.float32
        )

        return image, label, orig_size


def denormalize(img):
    """Reverse ImageNet normalization for visualization."""
    mean = torch.tensor(MEAN, dtype=img.dtype, device=img.device)
    std = torch.tensor(STD, dtype=img.dtype, device=img.device)

    mean = mean.view(3, 1, 1)
    std = std.view(3, 1, 1)

    return (img * std + mean).clamp(0, 1)


if __name__ == "__main__":
    root = "data/oxford-iiit-pet-noses"

    for split in ("train", "test"):
        dataset = PetNoseDataset(root=root, split=split)
        print(f"{split}: {len(dataset)} samples")

        if len(dataset) > 0:
            image, label, orig_size = dataset[0]
            print("  Image shape:", image.shape)
            print("  Label:", label)
            print("  Original size (W, H):", orig_size.tolist())
