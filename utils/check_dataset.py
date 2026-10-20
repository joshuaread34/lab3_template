from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from utils.dataset import PetNoseDataset, denormalize

DATASET_ROOT = "/content/drive/MyDrive/ELEC475/Lab3/data/oxford-iiit-pet-noses"
IMG_SIZE = 227
BATCH_SIZE = 64


def check_split(dataset, split):
    """Validate every sample using batches of 64."""
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    count = 0

    for images, labels, orig_sizes in loader:
        # Expected tensor dimensions and data types
        assert images.ndim == 4, f"{split}: incorrect image dimensions"
        assert images.shape[1:] == (3, IMG_SIZE, IMG_SIZE)
        assert labels.shape == (images.shape[0], 2)
        assert orig_sizes.shape == (images.shape[0], 2)
        assert images.dtype == torch.float32
        assert labels.dtype == torch.float32

        # Check that all values are finite
        assert torch.isfinite(images).all()
        assert torch.isfinite(labels).all()
        assert torch.isfinite(orig_sizes).all()

        # Nose coordinates should lie within the resized image.
        assert (labels[:, 0] >= 0).all()
        assert (labels[:, 0] < IMG_SIZE).all()
        assert (labels[:, 1] >= 0).all()
        assert (labels[:, 1] < IMG_SIZE).all()

        # Original image dimensions must be positive.
        assert (orig_sizes > 0).all()

        count += images.shape[0]

    assert count == len(dataset)

    print(f"{split}: passed validation for {count:,} samples")


def plot_examples(dataset):
    """Plot eight training images and their nose locations."""
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    for i, ax in enumerate(axes.flat):
        image, label, orig_size = dataset[i]

        # Undo normalization for display.
        image = denormalize(image)
        image = image.permute(1, 2, 0).cpu().numpy()

        u, v = label.tolist()

        ax.imshow(image)
        ax.scatter(u, v, c="red", s=45, marker="x")
        ax.set_title(f"Sample {i}: ({u:.1f}, {v:.1f})")
        ax.axis("off")

    fig.suptitle("Oxford-IIIT Pet Noses: Dataset Check")
    fig.tight_layout()

    output_path = Path("plots/dataset_check.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Saved plot to: {output_path.resolve()}")


def main():
    train_dataset = PetNoseDataset(
        root=DATASET_ROOT,
        split="train"
    )
    test_dataset = PetNoseDataset(
        root=DATASET_ROOT,
        split="test"
    )

    print(f"Training samples: {len(train_dataset):,}")
    print(f"Testing samples: {len(test_dataset):,}")

    check_split(train_dataset, "train")
    check_split(test_dataset, "test")

    plot_examples(train_dataset)


if __name__ == "__main__":
    main()
