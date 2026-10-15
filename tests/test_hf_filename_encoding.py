import struct

import pytest

from modelinfo.parsers import huggingface as hf


@pytest.mark.parametrize(("filename", "suffix"), [
    ("sub dir/model.safetensors", "sub%20dir/model.safetensors"),
    ("model#1.safetensors", "model%231.safetensors"),
    ("model?x.safetensors", "model%3Fx.safetensors"),
    ("model%20.safetensors", "model%2520.safetensors"),
])
def test_file_names_are_encoded_as_path_components(monkeypatch, filename, suffix):
    monkeypatch.setenv("HF_ENDPOINT", "https://hub.example")
    urls = []

    def request(url, **kwargs):
        urls.append(url)
        return struct.pack("<Q", 2) + b"{}"

    monkeypatch.setattr(hf, "_make_request", request)
    assert hf._fetch_safetensors_header("org/model", filename) == {}
    assert urls == ["https://hub.example/org/model/resolve/main/" + suffix]
