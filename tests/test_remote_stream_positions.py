import pytest

from modelinfo.parsers import huggingface as hf


def test_negative_seek_does_not_corrupt_position():
    stream = hf.RemoteFileStream("https://hub.example/file")
    assert stream.seek(3) == 3
    with pytest.raises(ValueError):
        stream.seek(-4, 1)
    assert stream.tell() == 3
    with pytest.raises(ValueError):
        stream.seek(-1)
    assert stream.tell() == 3


def test_invalid_negative_read_does_not_move_cursor():
    stream = hf.RemoteFileStream("https://hub.example/file")
    stream.buffer = b"abc"
    with pytest.raises(ValueError):
        stream.read(-2)
    assert stream.tell() == 0


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_invalid_chunk_size_rejected(chunk_size):
    with pytest.raises(ValueError):
        hf.RemoteFileStream("https://hub.example/file", chunk_size=chunk_size)


def test_chunked_reads_reuse_buffer_and_return_independent_bytes(monkeypatch):
    payload = b"abcdefghijkl"
    requests = []

    def request(url, headers, limit, timeout):
        requests.append((headers["Range"], limit, timeout))
        start = int(headers["Range"].split("=")[1].split("-")[0])
        return payload[start:start + limit]

    monkeypatch.setattr(hf, "_make_request", request)
    stream = hf.RemoteFileStream("https://hub.example/file", chunk_size=4, timeout=3)
    buffer = stream.buffer
    first = stream.read(5)
    assert first == b"abcde"
    assert isinstance(first, bytes)
    assert stream.buffer is buffer  # Growth must not replace the accumulated buffer.
    assert stream.read(5) == b"fghij"
    assert stream.buffer is buffer
    assert first == b"abcde"
    assert requests == [("bytes=0-3", 4, 3), ("bytes=4-7", 4, 3), ("bytes=8-11", 4, 3)]
    stream.seek(2)
    assert stream.read(7) == b"cdefghi"
    assert len(requests) == 3
    assert stream.tell() == 9


@pytest.mark.parametrize("range_error", [False, True])
def test_chunked_read_preserves_short_read_and_eof(monkeypatch, range_error):
    payload = b"abcdef"

    def request(url, headers, limit, timeout):
        start = int(headers["Range"].split("=")[1].split("-")[0])
        if start >= len(payload) and range_error:
            raise hf.urllib.error.HTTPError(url, 416, "range", {}, None)
        return payload[start:start + limit]

    monkeypatch.setattr(hf, "_make_request", request)
    stream = hf.RemoteFileStream("https://hub.example/file", chunk_size=4)
    assert stream.read(10) == payload
    assert stream.tell() == len(payload)
    assert stream.read(1) == b""
    assert stream.tell() == len(payload)


def test_read_limit_rejects_before_request_or_position_change(monkeypatch):
    def unexpected_request(*args, **kwargs):
        pytest.fail("over-limit read must not request data")

    monkeypatch.setattr(hf, "_make_request", unexpected_request)
    stream = hf.RemoteFileStream("https://hub.example/file")
    stream.seek(50 * 1024 * 1024)
    with pytest.raises(ValueError, match="50MB"):
        stream.read(1)
    assert stream.tell() == 50 * 1024 * 1024
    assert len(stream.buffer) == 0
