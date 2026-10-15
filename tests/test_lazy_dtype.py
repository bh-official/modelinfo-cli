import pytest

from modelinfo.calculator import calculate_footprint


@pytest.mark.parametrize('key,dtype,expected,width', [
    ('torch_dtype', 'float32', 'F32', 4),
    ('dtype', 'float64', 'F64', 8),
    ('torch_dtype', 'float16', 'F16', 2),
])
def test_lazy_parameter_estimate_respects_declared_dtype(key, dtype, expected, width):
    result = calculate_footprint({'__metadata__': {'lazy_fetch': True, 'total_size': 1024}}, config={key: dtype})
    assert result['primary_dtype'] == expected
    assert result['total_params'] == 1024 // width
    assert result['base_memory_bytes'] == 1024
