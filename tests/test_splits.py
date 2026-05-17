import numpy as np
import pytest

from galaxy_morph.data.splits import make_splits


@pytest.fixture
def balanced_labels() -> np.ndarray:
    # 200 samples, 2 classes, balanced
    return np.array([i % 2 for i in range(200)])


def test_split_sizes_match_fractions(balanced_labels: np.ndarray) -> None:
    n = len(balanced_labels)
    splits = make_splits(balanced_labels, val_frac=0.1, test_frac=0.1)

    assert len(splits.test) == pytest.approx(n * 0.1, abs=2)
    assert len(splits.val) == pytest.approx(n * 0.1, abs=2)
    assert len(splits.train) == pytest.approx(n * 0.8, abs=2)
    assert len(splits.train) + len(splits.val) + len(splits.test) == n


def test_splits_are_disjoint(balanced_labels: np.ndarray) -> None:
    splits = make_splits(balanced_labels)

    all_idx = np.concatenate([splits.train, splits.val, splits.test])
    assert len(all_idx) == len(np.unique(all_idx))


def test_splits_cover_all_indices(balanced_labels: np.ndarray) -> None:
    splits = make_splits(balanced_labels)

    all_idx = np.sort(np.concatenate([splits.train, splits.val, splits.test]))
    np.testing.assert_array_equal(all_idx, np.arange(len(balanced_labels)))


def test_split_is_reproducible(balanced_labels: np.ndarray) -> None:
    a = make_splits(balanced_labels, seed=42)
    b = make_splits(balanced_labels, seed=42)

    np.testing.assert_array_equal(a.train, b.train)
    np.testing.assert_array_equal(a.val, b.val)
    np.testing.assert_array_equal(a.test, b.test)


def test_different_seeds_produce_different_splits(balanced_labels: np.ndarray) -> None:
    a = make_splits(balanced_labels, seed=1)
    b = make_splits(balanced_labels, seed=2)

    assert not np.array_equal(a.train, b.train)
