import io

import pytest

from modelinfo.parsers import huggingface as hf


def test_nonzero_range_rejects_full_response(monkeypatch):
    monkeypatch.setattr(hf, "_get_hf_token", lambda: None)
    response = io.BytesIO(b"wrong prefix")
    response.status = 200
    monkeypatch.setattr(hf.urllib.request, "urlopen", lambda *a, **k: response)
    with pytest.raises(ValueError, match="range"):
        hf._make_request("https://hub.example/file", {"Range": "bytes=8-15"}, limit=8)


def test_nonzero_range_accepts_partial_response(monkeypatch):
    monkeypatch.setattr(hf, "_get_hf_token", lambda: None)
    response = io.BytesIO(b"expected")
    response.status = 206
    monkeypatch.setattr(hf.urllib.request, "urlopen", lambda *a, **k: response)
    assert hf._make_request("https://hub.example/file", {"Range": "bytes=8-15"}, limit=8) == b"expected"
