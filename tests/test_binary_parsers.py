"""Small generated checkpoint fixtures; no model downloads or torch dependency."""

import json
import struct
import zipfile
from pickle import UnpicklingError

import pytest

from modelinfo.parsers.pytorch import parse_pytorch_header
from modelinfo.parsers.safetensors import parse_safetensors_header


def write_safetensors(path, header):
    payload = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(payload)) + payload)


def test_safetensors_extracts_header_without_reading_tensor_payload(tmp_path):
    path = tmp_path / "model.safetensors"
    header = {
        "__metadata__": {"format": "pt"},
        "weight": {"shape": [2, 3], "dtype": "F16", "data_offsets": [0, 12]},
    }
    write_safetensors(path, header)
    assert parse_safetensors_header(str(path)) == header


@pytest.mark.parametrize("data", [b"", b"1234567", struct.pack("<Q", 20) + b"{}"])
def test_safetensors_rejects_truncated_headers(tmp_path, data):
    path = tmp_path / "truncated.safetensors"
    path.write_bytes(data)
    with pytest.raises(EOFError, match="Unexpected end of file"):
        parse_safetensors_header(str(path))


def test_safetensors_rejects_oversized_header_before_reading_it(tmp_path):
    path = tmp_path / "oversized.safetensors"
    path.write_bytes(struct.pack("<Q", 100 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="exceeds maximum allowed size"):
        parse_safetensors_header(str(path))


def test_safetensors_rejects_malformed_json(tmp_path):
    path = tmp_path / "invalid.safetensors"
    path.write_bytes(struct.pack("<Q", 1) + b"{")
    with pytest.raises(json.JSONDecodeError):
        parse_safetensors_header(str(path))


def test_safetensors_index_deduplicates_shards_and_reports_missing_files(tmp_path):
    index = tmp_path / "model.safetensors.index.json"
    shard = tmp_path / "part1.safetensors"
    header = {
        "__metadata__": {"format": "pt"},
        "a": {"shape": [2], "dtype": "F16"},
        "b": {"shape": [3], "dtype": "F32"},
    }
    write_safetensors(shard, header)
    index.write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": shard.name,
                    "b": shard.name,
                    "c": "missing.safetensors",
                }
            }
        )
    )
    result = parse_safetensors_header(str(index))
    assert result["a"] == header["a"]
    assert result["b"] == header["b"]
    assert "c" not in result
    assert result["__metadata__"] == {
        "missing_shards": 1,
        "total_shards": 2,
        "is_sharded": True,
        "disk_size": shard.stat().st_size,
    }


def test_pytorch_reads_ordered_state_dict_from_archive(tmp_path):
    path = tmp_path / "model.pt"
    # Fixed protocol-0 OrderedDict fixture: inspect metadata without generating
    # executable serialization during the test.
    state = b"ccollections\nOrderedDict\np0\n(tRp1\nVweight\np2\n(dp3\nVshape\np4\n(lp5\nI2\naI3\nasVdtype\np6\nVF16\np7\nssVstep\np8\nI12\ns."
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("checkpoint/data.pkl", state)
    assert parse_pytorch_header(str(path)) == {
        "weight": {"shape": [2, 3], "dtype": "F16"},
        "step": {"shape": [], "dtype": "F32"},
    }


def test_pytorch_rejects_non_zip_checkpoint(tmp_path):
    path = tmp_path / "legacy.pt"
    path.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="not a valid zip archive"):
        parse_pytorch_header(str(path))


def test_pytorch_rejects_archive_without_metadata(tmp_path):
    path = tmp_path / "empty.pt"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("checkpoint/version", b"3")
    with pytest.raises(ValueError, match="Could not find data.pkl"):
        parse_pytorch_header(str(path))


def test_pytorch_rejects_unapproved_pickle_global(tmp_path):
    path = tmp_path / "forbidden.pt"
    # A harmless builtin still must not bypass the explicit unpickler allowlist.
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("checkpoint/data.pkl", b"cbuiltins\nlen\n.")
    with pytest.raises(UnpicklingError, match="forbidden"):
        parse_pytorch_header(str(path))


def test_pytorch_propagates_truncated_pickle_error(tmp_path):
    path = tmp_path / "truncated.pt"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("checkpoint/data.pkl", b"")
    with pytest.raises(EOFError):
        parse_pytorch_header(str(path))
