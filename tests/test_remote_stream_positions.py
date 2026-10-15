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
