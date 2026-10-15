import pytest

from modelinfo.calculator import calculate_footprint


@pytest.mark.parametrize(("dtype", "size"), [("BOOL", 1), ("U8", 1), ("U16", 2)])
def test_unsigned_and_boolean_tensor_sizes(dtype, size):
    result = calculate_footprint({"buffer": {"dtype": dtype, "shape": [256]}})
    assert result["base_memory_bytes"] == 256 * size
