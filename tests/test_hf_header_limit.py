import struct

import pytest

from modelinfo.parsers import huggingface as hf


def test_remote_safetensors_rejects_oversized_header_before_second_request(monkeypatch):
    calls = []

    def request(*args, **kwargs):
        calls.append(kwargs.get("limit"))
        return struct.pack("<Q", 100 * 1024 * 1024 + 1)

    monkeypatch.setattr(hf, "_make_request", request)
    with pytest.raises(ValueError, match="maximum"):
        hf._fetch_safetensors_header("org/model", "model.safetensors")
    assert len(calls) == 1
