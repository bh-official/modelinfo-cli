"""Exercise HTTP request construction and checkpoint parsing entirely offline."""

import io
import json
import struct
import urllib.error

import pytest

from modelinfo.parsers import huggingface as hf


@pytest.fixture(autouse=True)
def isolated_hub(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://hub.example")
    monkeypatch.setattr(hf, "_get_hf_token", lambda: "test-token")


def response(data, **headers):
    stream = io.BytesIO(data)
    stream.headers = headers
    return stream


def test_request_sends_auth_range_timeout_and_limits_read(monkeypatch):
    calls = []

    def urlopen(request, timeout):
        calls.append((request, timeout))
        return response(b"0123456789")

    monkeypatch.setattr(hf.urllib.request, "urlopen", urlopen)
    assert (
        hf._make_request(
            "https://hub.example/file", {"Range": "bytes=0-3"}, limit=4, timeout=2.5
        )
        == b"0123"
    )
    request, timeout = calls[0]
    assert request.get_header("Authorization") == "Bearer test-token"
    assert request.get_header("Range") == "bytes=0-3"
    assert timeout == 2.5


@pytest.mark.parametrize(
    "status,error",
    [(401, PermissionError), (404, FileNotFoundError), (503, urllib.error.HTTPError)],
)
def test_request_translates_only_expected_http_errors(monkeypatch, status, error):
    def urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, status, "failure", {}, None)

    monkeypatch.setattr(hf.urllib.request, "urlopen", urlopen)
    with pytest.raises(error):
        hf._make_request("https://hub.example/file")


def test_single_checkpoint_uses_api_head_and_bounded_header_request(monkeypatch):
    header = {"weight": {"shape": [2, 3], "dtype": "F16"}}
    encoded = json.dumps(header).encode()
    calls = []

    def urlopen(request, timeout):
        calls.append((request.get_method(), request.full_url, timeout))
        assert request.get_header("Authorization") == "Bearer test-token"
        if "/api/models/" in request.full_url:
            return response(
                json.dumps({"siblings": [{"rfilename": "model.safetensors"}]}).encode()
            )
        if request.get_method() == "HEAD":
            return response(b"", **{"Content-Length": "128"})
        assert request.get_header("Range") == "bytes=0-500000"
        return response(struct.pack("<Q", len(encoded)) + encoded)

    monkeypatch.setattr(hf.urllib.request, "urlopen", urlopen)
    assert hf.fetch_huggingface_repo("org/model", timeout=3.5) == (
        header,
        None,
        "SafeTensors",
        128.0,
    )
    assert [method for method, _, _ in calls] == ["GET", "HEAD", "GET"]
    assert all(timeout == 3.5 for _, _, timeout in calls)


def test_large_header_fetches_only_remaining_header_range(monkeypatch):
    encoded = json.dumps({"__metadata__": {"note": "x" * 500000}}).encode()
    calls = []

    def urlopen(request, timeout):
        range_header = request.get_header("Range")
        calls.append(range_header)
        if len(calls) == 1:
            return response(struct.pack("<Q", len(encoded)) + encoded[:499992])
        assert range_header == f"bytes=8-{len(encoded) + 7}"
        return response(encoded)

    monkeypatch.setattr(hf.urllib.request, "urlopen", urlopen)
    assert hf._fetch_safetensors_header("org/model", "model.safetensors") == json.loads(
        encoded
    )
    assert len(calls) == 2


def test_unsatisfiable_range_retries_without_range_header(monkeypatch):
    encoded = b"{}"
    calls = []

    def urlopen(request, timeout):
        calls.append(request.get_header("Range"))
        if len(calls) == 1:
            raise urllib.error.HTTPError(request.full_url, 416, "range", {}, None)
        return response(struct.pack("<Q", len(encoded)) + encoded)

    monkeypatch.setattr(hf.urllib.request, "urlopen", urlopen)
    assert hf._fetch_safetensors_header("org/model", "model.safetensors") == {}
    assert calls == ["bytes=0-500000", None]


def test_lazy_sharded_metadata_never_downloads_weights(monkeypatch):
    config = {"hidden_size": 8, "num_hidden_layers": 2}
    index = {
        "metadata": {"total_size": 128},
        "weight_map": {
            "a": "part1.safetensors",
            "b": "part1.safetensors",
            "c": "part2.safetensors",
        },
    }
    calls = []

    def urlopen(request, timeout):
        url = request.full_url
        calls.append(url)
        if "/api/models/" in url:
            return response(
                json.dumps(
                    {
                        "siblings": [
                            {"rfilename": "config.json"},
                            {"rfilename": "model.safetensors.index.json"},
                        ]
                    }
                ).encode()
            )
        if url.endswith("/config.json"):
            return response(json.dumps(config).encode())
        if url.endswith("/model.safetensors.index.json"):
            return response(json.dumps(index).encode())
        pytest.fail(f"Unexpected weight download: {url}")

    monkeypatch.setattr(hf.urllib.request, "urlopen", urlopen)
    tensors, actual_config, fmt, size = hf.fetch_huggingface_repo("org/model")
    assert actual_config == config
    assert (fmt, size) == ("SafeTensors", 128.0)
    assert set(tensors) == {"a", "b", "c", "__metadata__"}
    assert tensors["__metadata__"]["lazy_fetch"] is True
    assert tensors["__metadata__"]["total_shards"] == 2
    assert len(calls) == 3


def test_remote_stream_reuses_buffer_and_stops_at_eof(monkeypatch):
    payload = b"abcdefghij"
    calls = []

    def urlopen(request, timeout):
        start, end = map(
            int, request.get_header("Range").removeprefix("bytes=").split("-")
        )
        calls.append((start, end))
        if start >= len(payload):
            raise urllib.error.HTTPError(request.full_url, 416, "range", {}, None)
        return response(payload[start : end + 1])

    monkeypatch.setattr(hf.urllib.request, "urlopen", urlopen)
    stream = hf.RemoteFileStream("https://hub.example/file", chunk_size=4)
    assert stream.read(3) == b"abc"
    assert stream.seek(1) == 1
    assert stream.read(3) == b"bcd"
    assert calls == [(0, 3)]
    assert stream.read(20) == b"efghij"
    assert stream.tell() == 10
    assert stream.read(1) == b""


def test_request_does_not_retain_token_in_reused_headers(monkeypatch):
    calls = []
    def urlopen(request, timeout):
        calls.append(request)
        return response(b"ok")
    monkeypatch.setattr(hf.urllib.request, "urlopen", urlopen)
    headers = {"Range": "bytes=0-1"}
    hf._make_request("https://hub.example/file", headers=headers)
    monkeypatch.setattr(hf, "_get_hf_token", lambda: None)
    hf._make_request("https://hub.example/file", headers=headers)
    assert calls[0].get_header("Authorization") == "Bearer test-token"
    assert calls[1].get_header("Authorization") is None
    assert headers == {"Range": "bytes=0-1"}
