"""
ELEC 475: Lab 3 - Localization Metrics & Mean-Location Baseline (Starter Template)
Provides:
  1. euclidean_errors: Per-image localization error in original-image pixels.
  2. error_stats: Min, mean, median, max and standard deviation of the errors.
  3. baseline_predictions: Mean-location baseline built from the training labels.
  4. ced_curve: Points of the cumulative error distribution (CED) curve.
  5. best_worst_indices: Indices of the best and worst predictions.
  6. print_results: Accuracy and cost tables (provided).
"""
import os
from typing import Dict, Tuple
import torch
from PIL import Image


def euclidean_errors(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    differences = pred - gt
    return torch.sqrt((differences ** 2).sum(dim=1))


def error_stats(errors: torch.Tensor) -> Dict[str, float]:
    return {
        "min": errors.min().item(),
        "mean": errors.mean().item(),
        "median": errors.median().item(),
        "max": errors.max().item(),
        "std": errors.std().item(),
    }


def baseline_predictions(orig_size: torch.Tensor, train_file: str, img_dir: str) -> torch.Tensor:
    normalized_positions = []

    with open(train_file, 'r') as f:
        entries = [line.strip() for line in f if line.strip()]

    print(f'Processing {len(entries)} training images...', flush=True)

    for i, line in enumerate(entries):
        name, uv = line.split(',', 1)
        u, v = map(float, uv.strip().strip(chr(34)).strip('()').split(','))
        with Image.open(os.path.join(img_dir, name.strip())) as img:
            width, height = img.size
        normalized_positions.append([u / width, v / height])
        if (i + 1) % 500 == 0:
            print(f'Processed {i + 1}/{len(entries)}', flush=True)

    avg_position = torch.tensor(normalized_positions, dtype=torch.float32).mean(dim=0)
    predictions = torch.empty_like(orig_size, dtype=torch.float32)
    predictions[:, 0] = avg_position[0] * orig_size[:, 0]
    predictions[:, 1] = avg_position[1] * orig_size[:, 1]
    return predictions

def ced_curve(errors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.sort(errors).values
    n = len(errors)
    y = 100.0 * torch.arange(1, n + 1, dtype=errors.dtype, device=errors.device) / n
    return x, y


def best_worst_indices(errors: torch.Tensor, k: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    indices = torch.argsort(errors)
    best = indices[:k]
    worst = indices[-k:].flip(0)
    return best, worst


def print_results(rows: Dict[str, Dict]):
    """
    Provided. Prints the accuracy and cost tables.

    rows : dict mapping each model name to a dict with the keys
           "errors" (tensor of errors), "params" (int) and "ms" (float)
    """
    stats = {name: error_stats(row["errors"]) for name, row in rows.items()}
    w = max([19] + [len(name) + 2 for name in rows])     # name column width
    print()
    if all(s is not None for s in stats.values()):
        print(f"{'Model':<{w}}" + "".join(f"{h:>8}" for h in ["Min", "Mean", "Median", "Max", "Std"]))
        for name, s in stats.items():
            print(f"{name:<{w}}" + "".join(f"{s[k]:>8.1f}" for k in ["min", "mean", "median", "max", "std"]))
        print()
    else:
        print("[TODO] error_stats() is not implemented yet: accuracy table skipped\n")
    print(f"{'Model':<{w}}{'Params':>13}{'ms/img':>9}")
    for name, row in rows.items():
        print(f"{name:<{w}}{row['params']:>13,}{row['ms']:>9.1f}")
    print()
