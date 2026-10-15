from unittest.mock import patch

from modelinfo.parsers.huggingface import RemoteFileStream


def test_zero_length_read_after_seek_never_fetches_or_moves():
    stream = RemoteFileStream('https://example.test/model.gguf')
    for position in (100, 60 * 1024 * 1024):
        stream.seek(position)
        with patch('modelinfo.parsers.huggingface._make_request', side_effect=AssertionError('unexpected network request')):
            assert stream.read(0) == b''
        assert stream.tell() == position
