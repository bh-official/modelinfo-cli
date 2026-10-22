"""Local model directory auto-selection (issue #58)."""

import json
import shutil
from pathlib import Path

import pytest

from modelinfo import cli

FIXTURES = Path(__file__).parent / "fixtures"


def _copy_fixture(name: str, dest: Path) -> Path:
    target = dest / name
    shutil.copy(FIXTURES / name, target)
    return target


def _fake_footprint(*args, **kwargs):
    return {
        "total_params": 1,
        "base_memory_bytes": 2.0,
        "kv_cache_bytes": 1.0,
        "overhead_bytes": 0.0,
        "total_memory_bytes": 3.0,
        "num_layers": 1,
        "kv_dim": 1,
        "primary_dtype": "BF16",
        "kv_is_estimate": False,
        "penalty_percentage": 0.0,
        "vllm_metrics": {},
    }


def test_directory_with_safetensors_index_resolves_to_index(tmp_path, monkeypatch):
    model_dir = tmp_path / "Mistral-7B-v0.1"
    model_dir.mkdir()
    shard = _copy_fixture("mock_mistral-7b.safetensors", model_dir)
    index = model_dir / "model.safetensors.index.json"
    index.write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": shard.name}})
    )
    shutil.copy(FIXTURES / "config.json", model_dir / "config.json")

    captured = {}

    def fake_parse_safetensors_header(file_path):
        captured["path"] = file_path
        return {
            "__metadata__": {"is_sharded": True, "disk_size": 10},
            "model.embed_tokens.weight": {
                "shape": [1, 1],
                "dtype": "BF16",
                "data_offsets": [0, 2],
            },
        }

    monkeypatch.setattr(cli, "parse_safetensors_header", fake_parse_safetensors_header)
    monkeypatch.setattr(cli, "calculate_footprint", _fake_footprint)
    monkeypatch.setattr(cli, "identify_architecture_name", lambda *a, **k: "Mock")

    info = cli.analyze_model(str(model_dir), context_override=128)

    assert Path(captured["path"]) == index
    assert info["format_name"] == "SafeTensors"
    assert info["tensors"]["__metadata__"].get("is_sharded") is True


def test_directory_with_single_safetensors_resolves_to_file(tmp_path, monkeypatch):
    model_dir = tmp_path / "single-st"
    model_dir.mkdir()
    weights = _copy_fixture("mock_mistral-7b.safetensors", model_dir)
    shutil.copy(FIXTURES / "config.json", model_dir / "config.json")

    captured = {}

    def fake_parse_safetensors_header(file_path):
        captured["path"] = file_path
        return {
            "weight": {"shape": [1, 1], "dtype": "BF16", "data_offsets": [0, 2]},
        }

    monkeypatch.setattr(cli, "parse_safetensors_header", fake_parse_safetensors_header)
    monkeypatch.setattr(cli, "calculate_footprint", _fake_footprint)
    monkeypatch.setattr(cli, "identify_architecture_name", lambda *a, **k: "Mock")

    info = cli.analyze_model(str(model_dir), context_override=128)

    assert Path(captured["path"]) == weights
    assert info["format_name"] == "SafeTensors"
    assert info["tensors"].get("weight") is not None


def test_directory_with_multiple_gguf_returns_group(tmp_path, monkeypatch):
    model_dir = tmp_path / "gguf-repo"
    model_dir.mkdir()
    q4 = model_dir / "model-Q4_K_M.gguf"
    q8 = model_dir / "model-Q8_0.gguf"
    shutil.copy(FIXTURES / "mock_model.gguf", q4)
    shutil.copy(FIXTURES / "mock_model.gguf", q8)
    # Make sizes differ so the group picks a stable header target.
    q8.write_bytes(q8.read_bytes() + b"\x00" * 64)

    def fake_parse_gguf_header(file_path):
        return {
            "__metadata__": {
                "general.architecture": "llama",
                "llama.context_length": 4096,
                "llama.block_count": 32,
            },
            "blk.0.attn_q.weight": {"shape": [1, 1], "dtype": "Q4_0"},
        }

    def fake_footprint(*args, **kwargs):
        fp = _fake_footprint()
        fp["num_layers"] = 32
        fp["primary_dtype"] = "Q4_0"
        return fp

    monkeypatch.setattr(cli, "parse_gguf_header", fake_parse_gguf_header)
    monkeypatch.setattr(cli, "calculate_footprint", fake_footprint)
    monkeypatch.setattr(cli, "identify_architecture_name", lambda *a, **k: "Llama")

    info = cli.analyze_model(str(model_dir), context_override=128)

    assert info["format_name"] == "GGUF_group"
    variants = info["tensors"]["__metadata__"]["gguf_variants"]
    names = {v["filename"] for v in variants}
    assert names == {"model-Q4_K_M.gguf", "model-Q8_0.gguf"}
    assert info["tensors"]["__metadata__"]["repo_id"] == str(model_dir)


def test_directory_with_single_gguf_resolves_to_file(tmp_path, monkeypatch):
    model_dir = tmp_path / "one-gguf"
    model_dir.mkdir()
    weights = _copy_fixture("mock_model.gguf", model_dir)

    captured = {}

    def fake_parse_gguf_header(file_path):
        captured["path"] = file_path
        return {
            "__metadata__": {"general.architecture": "llama"},
            "blk.0.attn_q.weight": {"shape": [1, 1], "dtype": "F16"},
        }

    monkeypatch.setattr(cli, "parse_gguf_header", fake_parse_gguf_header)
    monkeypatch.setattr(cli, "calculate_footprint", _fake_footprint)
    monkeypatch.setattr(cli, "identify_architecture_name", lambda *a, **k: "Mock")

    info = cli.analyze_model(str(model_dir), context_override=128)

    assert Path(captured["path"]) == weights
    assert info["format_name"] == "GGUF"


def test_directory_prefers_pytorch_model_bin(tmp_path, monkeypatch):
    model_dir = tmp_path / "pt-dir"
    model_dir.mkdir()
    preferred = model_dir / "pytorch_model.bin"
    other = model_dir / "extra.pt"
    shutil.copy(FIXTURES / "mock_model.pt", preferred)
    shutil.copy(FIXTURES / "mock_model.pt", other)

    captured = {}

    def fake_parse_pytorch_header(file_path):
        captured["path"] = file_path
        return {"weight": {"shape": [1, 1], "dtype": "F32"}}

    monkeypatch.setattr(cli, "parse_pytorch_header", fake_parse_pytorch_header)
    monkeypatch.setattr(cli, "calculate_footprint", _fake_footprint)
    monkeypatch.setattr(cli, "identify_architecture_name", lambda *a, **k: "Mock")

    info = cli.analyze_model(str(model_dir), context_override=128)

    assert Path(captured["path"]) == preferred
    assert info["format_name"] == "PyTorch"


def test_directory_with_single_pt_resolves_to_file(tmp_path, monkeypatch):
    model_dir = tmp_path / "one-pt"
    model_dir.mkdir()
    weights = _copy_fixture("mock_model.pt", model_dir)

    captured = {}

    def fake_parse_pytorch_header(file_path):
        captured["path"] = file_path
        return {"weight": {"shape": [1, 1], "dtype": "F32"}}

    monkeypatch.setattr(cli, "parse_pytorch_header", fake_parse_pytorch_header)
    monkeypatch.setattr(cli, "calculate_footprint", _fake_footprint)
    monkeypatch.setattr(cli, "identify_architecture_name", lambda *a, **k: "Mock")

    info = cli.analyze_model(str(model_dir), context_override=128)

    assert Path(captured["path"]) == weights
    assert info["format_name"] == "PyTorch"


def test_directory_with_no_weights_raises(tmp_path):
    model_dir = tmp_path / "empty-model"
    model_dir.mkdir()
    (model_dir / "README.md").write_text("no weights here")

    with pytest.raises(ValueError, match="No supported weight files"):
        cli.analyze_model(str(model_dir), context_override=128)


def test_directory_with_multiple_safetensors_without_index_lists_candidates(tmp_path):
    model_dir = tmp_path / "multi-st"
    model_dir.mkdir()
    a = model_dir / "a.safetensors"
    b = model_dir / "b.safetensors"
    shutil.copy(FIXTURES / "mock_mistral-7b.safetensors", a)
    shutil.copy(FIXTURES / "mock_mistral-7b.safetensors", b)

    with pytest.raises(ValueError, match="Multiple .safetensors files") as exc_info:
        cli.analyze_model(str(model_dir), context_override=128)

    message = str(exc_info.value)
    assert "a.safetensors" in message
    assert "b.safetensors" in message
