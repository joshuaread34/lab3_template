"""
ELEC 475: Lab 3 - Correction Model Dataset (Starter Template)
Builds training patches for the correction model from the training set only.

CorrectionDataset is a subclass of your Pre-Lab PetNoseDataset: it reuses its
parsed labels (self.samples) and image folder (self.img_dir), and replaces only
__getitem__.

Each sample is a 64 x 64 patch cut from a 128 x 128 region of the original image,
centred on a point near the true nose. The target is the offset from the patch
centre to the true nose, divided by HALF_PATCH so it lies roughly in [-1, 1]:

    corrected (u, v) = patch centre + target * HALF_PATCH
"""
import random
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from utils.dataset import PetNoseDataset, MEAN, STD

PATCH_SIZE = 128              # side of the cropped region, in original-image pixels
HALF_PATCH = PATCH_SIZE // 2  # offsets are divided by this to normalize them
INPUT_SIZE = 64               # the patch is resized to INPUT_SIZE x INPUT_SIZE
MAX_OFFSET = 40               # largest training offset in u and v, in original-image pixels


def extract_patch(img: Image.Image, center) -> Image.Image:
    """
    Extract a PATCH_SIZE square around center and resize it to INPUT_SIZE.
    Areas outside the original image are filled with black.
    """
    cx, cy = int(center[0]), int(center[1])
    half = PATCH_SIZE // 2

    left = cx - half
    top = cy - half
    right = left + PATCH_SIZE
    bottom = top + PATCH_SIZE

    patch = img.crop((left, top, right, bottom))
    patch = patch.resize(
        (INPUT_SIZE, INPUT_SIZE),
        resample=Image.BILINEAR
    )

    return patch


class CorrectionDataset(PetNoseDataset):
    """
    One random patch per training image per epoch. Inherits self.samples
    (a list of (name, u, v)), self.img_dir and __len__ from PetNoseDataset.
    """
    def __init__(self, root: str = "data/oxford-iiit-pet-noses", augment: bool = True):
        super(CorrectionDataset, self).__init__(root=root, split="train")
        self.augment = augment
        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)

    def __getitem__(self, idx: int):
        """
        Returns a normalized image patch and the normalized nose offset.
        """
        name, u, v = self.samples[idx]

        # Open the original image and ensure it is RGB.
        with Image.open(self.img_dir / name if not isinstance(self.img_dir, str)
                        else __import__("os").path.join(self.img_dir, name)) as source:
            img = source.convert("RGB")

        # Choose a random patch centre near the true nose.
        cx = round(u + random.uniform(-MAX_OFFSET, MAX_OFFSET))
        cy = round(v + random.uniform(-MAX_OFFSET, MAX_OFFSET))

        # Offset from patch centre to the true nose, in original pixels.
        du = float(u) - cx
        dv = float(v) - cy

        patch = extract_patch(img, (cx, cy))

        if self.augment:
            # Horizontal flipping reverses the horizontal offset.
            if random.random() < 0.5:
                patch = patch.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                du = -du

            # Colour jitter changes appearance, not nose coordinates.
            patch = self.color_jitter(patch)

        patch = TF.normalize(TF.to_tensor(patch), MEAN, STD)
        target = torch.tensor(
            [du / HALF_PATCH, dv / HALF_PATCH],
            dtype=torch.float32
        )

        return patch, target



if __name__ == "__main__":
    # Reality check: saves 8 patches with their centre (red cross) and
    # target nose position (green circle) drawn on.
    import os
    import matplotlib.pyplot as plt
    from utils.dataset import denormalize
    from utils.visualization import show_prediction

    centre = INPUT_SIZE / 2
    scale = INPUT_SIZE / PATCH_SIZE               # original-image pixels -> patch pixels
    os.makedirs("plots", exist_ok=True)
    try:
        for augment in [False, True]:
            ds = CorrectionDataset(augment=augment)
            random.seed(0)
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            for ax, i in zip(axes.flat, range(8)):
                patch, target = ds[i]
                offset = target * HALF_PATCH
                nose = (centre + offset[0] * scale, centre + offset[1] * scale)
                show_prediction(ax, TF.to_pil_image(denormalize(patch)), nose, (centre, centre),
                                f"offset ({offset[0]:.0f}, {offset[1]:.0f}) px")
            save_path = os.path.join("plots", f"correction_patches{'_aug' if augment else ''}.png")
            fig.tight_layout()
            fig.savefig(save_path, dpi=200)
            plt.close(fig)
            print(f"{len(ds):,} training images | augment={augment} | saved to: {save_path}")
    except Exception as e:
        print(f"CorrectionDataset not fully implemented yet: {e}")
