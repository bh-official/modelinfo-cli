"""Rendered output checks using Rich's in-memory console."""

import io

import pytest
from rich.console import Console

from modelinfo import ui


@pytest.fixture
def render(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(
        ui, "console", Console(file=stream, width=160, color_system=None)
    )

    def show(**overrides):
        args = {
            "format_name": "SafeTensors",
            "arch_name": "Test model",
            "tensor_count": 1,
            "footprint": {
                "total_params": 6,
                "primary_dtype": "F16",
                "base_memory_bytes": 12,
                "kv_cache_bytes": 1024,
                "overhead_bytes": 2048,
                "total_memory_bytes": 3084,
            },
            "disk_size": 100,
            "context_length": 128,
            "is_default_context": False,
            "tensors": {"weight": {"shape": [2, 3], "dtype": "F16"}},
        }
        args.update(overrides)
        ui.print_model_info(**args)
        return stream.getvalue()

    return show


@pytest.mark.parametrize("missing", [False, True])
def test_incomplete_model_with_gpu_reports_unknown_instead_of_crashing(render, missing):
    tensors = (
        {"__metadata__": {"missing_shards": 1, "total_shards": 2}} if missing else {}
    )
    text = render(
        tensors=tensors,
        gpu_name="Test GPU",
        footprint={
            "total_params": 0,
            "primary_dtype": "Unknown",
            "total_memory_bytes": 0,
        },
    )
    assert "Hardware Fit:" in text
    assert "Unknown" in text
    assert "Fits comfortably" not in text
    if missing:
        assert "Missing 1 of 2 shards" in text
    else:
        assert "Missing Tensor Shapes" in text


def test_complete_model_renders_memory_breakdown_and_target_fit(render):
    text = render(gpu_name="Test GPU")
    for expected in [
        "Test model",
        "VRAM (est):",
        "Weights:",
        "KV Cache:",
        "Overhead:",
        "Fits comfortably",
        "Top Tensors by Size:",
        "weight",
    ]:
        assert expected in text


def test_lazy_model_omits_tensor_table_and_warns_about_context(render):
    text = render(is_lazy=True, context_length=1024, max_context=512)
    assert "exceeds model" in text
    assert "Run with --tensors" in text
    assert "Top Tensors by Size:" not in text


@pytest.mark.parametrize("size,expected", [(8, "green"), (16, "yellow"), (17, "red")])
def test_vram_color_boundaries(size, expected):
    assert ui.get_vram_color(size * 1024**3, 8) == expected


def test_tensor_groups_exclude_metadata_and_combine_repeated_layers():
    tensors = {
        "__metadata__": {"format": "pt"},
        "layers.0.weight": {"shape": [2, 3], "dtype": "F16"},
        "layers.1.weight": {"shape": [2, 3], "dtype": "F16"},
        "bias": {"shape": [2], "dtype": "F16"},
    }
    groups = ui.group_tensors_by_size(tensors)
    assert groups == [
        (("layers.[N].weight", (2, 3), "F16"), {"count": 2, "params": 6}),
        (("bias", (2,), "F16"), {"count": 1, "params": 2}),
    ]
