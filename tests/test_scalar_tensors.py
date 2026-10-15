from modelinfo.calculator import calculate_footprint


def test_scalar_tensor_contributes_one_parameter():
    result = calculate_footprint({"scale": {"shape": [], "dtype": "F32"}})
    assert result["total_params"] == 1
    assert result["base_memory_bytes"] == 4
    assert result["primary_dtype"] == "F32"


def test_missing_shape_is_not_assumed_scalar():
    result = calculate_footprint({"unknown": {"dtype": "F32"}})
    assert result["total_params"] == 0


def test_zero_length_tensor_stays_empty():
    result = calculate_footprint({"empty": {"shape": [0, 3], "dtype": "F32"}})
    assert result["total_params"] == 0
    assert result["base_memory_bytes"] == 0
