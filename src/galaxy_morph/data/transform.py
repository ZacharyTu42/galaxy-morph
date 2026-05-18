"""Data augmentation for galaxy images."""

from __future__ import annotations

import torchvision.transforms.v2 as T  # noqa: N812

GALAXY10_MEAN = (0.16747765243053436, 0.16261132061481476, 0.15892967581748962)
GALAXY10_STD = (0.12869802117347717, 0.11804956942796707, 0.11161898076534271)


def build_train_transform() -> T.Transform:
    """Augmentation pipeline for training.

    Galaxy images are rotation- and reflection-invariant, so we exploit
    that with random rotations from [-180, 180) and random flips.
    """
    return T.Compose(
        [
            T.RandomRotation(degrees=(-180, 180), interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.Normalize(mean=GALAXY10_MEAN, std=GALAXY10_STD),
        ]
    )


def build_val_transform(image_size: int = 224) -> T.Transform:
    """Augmentation pipeline for validation/testing.

    Only normalization, no random augmentations.
    """
    return T.Compose(
        [
            T.CenterCrop(image_size),
            T.Normalize(mean=GALAXY10_MEAN, std=GALAXY10_STD),
        ]
    )
