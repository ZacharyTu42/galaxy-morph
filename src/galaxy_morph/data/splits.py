"""Train/val/test split utilities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

@dataclass(frozen=True)
class DataSplits:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    
def make_splits(
        labels: np.ndarray,
        *,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 2000,
) -> DataSplits:
    """"Stratified train/val/test split.
    
    Returns indices into the original dataset, not the data itself.
    Stratification ensures class proportions are preserved across splits.
    """
    n = len(labels)
    indices = np.arange(n)

    # First split off the test set
    trainval_idx, test_idx = train_test_split(
        indices,
        test_size=test_frac,
        stratify=labels,
        random_state=seed,
    )

    # Splits train/val from remaining data
    relative_val = val_frac / (1 - test_frac)
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=relative_val,
        stratify=labels[trainval_idx],
        random_state=seed,
    )

    return DataSplits(
        train=np.sort(train_idx),
        val=np.sort(val_idx),
        test=np.sort(test_idx),
    )