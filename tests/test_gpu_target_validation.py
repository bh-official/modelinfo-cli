import pytest

from modelinfo.hardware import resolve_gpu


@pytest.mark.parametrize(
    "target", ["0x RTX4090", "0x 16", "0", "-1", "nan", "inf", "2x nan", "2x -16"]
)
def test_gpu_targets_reject_nonpositive_or_nonfinite_capacity(target):
    with pytest.raises(ValueError):
        resolve_gpu(target)


@pytest.mark.parametrize(
    "target,capacity,count", [("0.5", 0.5, 1), ("2x 12", 24, 2), ("RTX-4090", 24, 1)]
)
def test_gpu_target_validation_preserves_valid_names_and_capacities(
    target, capacity, count
):
    _, actual_capacity, actual_count = resolve_gpu(target)
    assert (actual_capacity, actual_count) == (capacity, count)
