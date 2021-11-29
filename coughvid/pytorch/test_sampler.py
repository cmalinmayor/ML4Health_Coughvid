import pytest
from coughvid.pytorch import SubsetWeightedRandomSampler, compute_weights

def test_compute_weights():
    labels = [0, 1, 1, 1, 0, 1, 1, 1]
    subset_indices = [0, 1, 2, 3]
    expected_weights = [1, 1/3, 1/3, 1/3]
    weights = compute_weights(labels, subset_indices)
    assert (expected_weights == weights).all()


def test_sampler():
    labels = [0, 1, 1, 1, 0, 1, 1, 1]
    subset_indices = [0, 1, 2, 3]
    weights = compute_weights(labels, subset_indices)
    num_samples=10
    replacement=True
    sampler = SubsetWeightedRandomSampler(
            subset_indices, weights, num_samples)

    indices = list(iter(sampler))
    print(indices)
    assert len(indices) == num_samples
    for index in indices:
        assert index in list(range(4))
