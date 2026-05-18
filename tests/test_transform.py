import pytest
import torch

from galaxy_morph.data.transform import (
    GALAXY10_MEAN,
    GALAXY10_STD,
    build_train_transform,
    build_val_transform,
)


@pytest.fixture
def test_image() -> torch.Tensor:
    # Create a test image with shape (3, 256, 256) and values in [0, 1]
    torch.manual_seed(42)
    return torch.rand(3, 256, 256)


def test_train_transform_shape(test_image: torch.Tensor) -> None:
    transform = build_train_transform()
    transformed = transform(test_image)

    # Check if shape is unchanged
    assert transformed.shape == test_image.shape


def test_train_transform_stochasticity(test_image: torch.Tensor) -> None:
    transform = build_train_transform()
    transformed1 = transform(test_image)
    transformed2 = transform(test_image)

    # With random augmentations, the two transformed images should not be identical
    assert not torch.equal(transformed1, transformed2)


def test_val_transform_center_crop(test_image: torch.Tensor) -> None:
    transform = build_val_transform(image_size=224)
    transformed = transform(test_image)

    # Check if the center crop was applied correctly
    assert transformed.shape == (3, 224, 224)


def test_val_transform_normalization(test_image: torch.Tensor) -> None:
    transform = build_val_transform(image_size=256)
    transformed = transform(test_image)

    # Check if the transformed image has zero mean and unit std (approximately)
    mean = transformed.mean(dim=[1, 2])
    std = transformed.std(dim=[1, 2])

    mean_expected = (test_image.mean(dim=[1, 2]) - torch.tensor(GALAXY10_MEAN)) / torch.tensor(
        GALAXY10_STD
    )
    std_expected = test_image.std(dim=[1, 2]) / torch.tensor(GALAXY10_STD)

    assert torch.allclose(mean, mean_expected, atol=0.1)
    assert torch.allclose(std, std_expected, atol=0.1)


def test_val_transform_no_augmentation(test_image: torch.Tensor) -> None:
    transform = build_val_transform()
    transformed1 = transform(test_image)
    transformed2 = transform(test_image)

    # With no random augmentations, the two transformed images should be identical
    assert torch.equal(transformed1, transformed2)
