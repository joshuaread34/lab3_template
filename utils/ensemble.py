"""
ELEC 475: Lab 3 - Prediction Ensembles (Starter Template)
Combines the test-set predictions of several models into one prediction per image.
"""
from typing import List
import torch


def ensemble_mean(preds: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack(preds, dim=0).mean(dim=0)


def ensemble_median(preds: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack(preds, dim=0).median(dim=0).values
