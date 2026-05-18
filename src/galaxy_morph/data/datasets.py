"""PyTorch Dataset for Galaxy10 DECaLS."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from galaxy_morph.data.splits import DataSplits
from galaxy_morph.data.transform import build_train_transform, build_val_transform

GALAXY10_CLASSES = [
    "disturbed",
    "merging",
    "round_smooth",
    "in_between_smooth",
    "cigar_smooth",
    "barred_spiral",
    "unbarred_tight_spiral",
    "unbarred_loose_spiral",
    "edge_on_no_bulge",
    "edge_on_with_bulge",
]


class Galaxy10Dataset(Dataset[tuple[torch.Tensor, int]]):
    """Galaxy10 DECaLS dataset wrapping the HDF5 file.

    Images are stored as uint8 (H, W, C) and converted to float32
    (C, H, W) tensors in __getitem__. The HDF5 file is opened lazily per
    worker to avoid multiprocessing issues.
    """

    def __init__(
        self,
        h5_path: Path | str,
        indices: np.ndarray | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        # Load labels and indices from the HDF5 file
        with h5py.File(self.h5_path, "r") as f:
            labels_ds = f["ans"]
            assert isinstance(labels_ds, h5py.Dataset)
            all_labels = labels_ds[:].astype(np.int64)

            self.indices = indices if indices is not None else np.arange(len(all_labels))
            self.labels = all_labels[self.indices]
            self.transform = transform

            # File handle is opened lazily per worker
            self._h5_file: h5py.File | None = None

    def _get_file(self) -> h5py.File:
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        true_idx = self.indices[idx]
        f = self._get_file()

        # Read image as uint8
        images_ds = f["images"]
        assert isinstance(images_ds, h5py.Dataset)
        img = images_ds[true_idx]  # shape (H, W, C), dtype uint8
        label = int(self.labels[idx])

        # Convert to float32 tensor (C, H, W)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @property
    def num_classes(self) -> int:
        return len(GALAXY10_CLASSES)

    @staticmethod
    def class_names(label: int) -> str:
        return GALAXY10_CLASSES[label]


def make_dataloaders(
    h5_path: Path,
    splits: DataSplits,
    *,
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = 224,
) -> tuple[
    DataLoader[tuple[torch.Tensor, int]],
    DataLoader[tuple[torch.Tensor, int]],
    DataLoader[tuple[torch.Tensor, int]],
]:
    """Create DataLoaders for train/val/test splits."""
    train_ds = Galaxy10Dataset(
        h5_path, indices=splits.train, transform=build_train_transform(image_size)
    )
    val_ds = Galaxy10Dataset(h5_path, indices=splits.val, transform=build_val_transform(image_size))
    test_ds = Galaxy10Dataset(
        h5_path, indices=splits.test, transform=build_val_transform(image_size)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, test_loader
