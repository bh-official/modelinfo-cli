import io
import struct

import pytest

from modelinfo.parsers.gguf import parse_gguf_header


def tensor_header():
    return (b'GGUF' + struct.pack('<IQQ', 3, 1, 0)
            + struct.pack('<Q', 6) + b'weight'
            + struct.pack('<IQIQ', 1, 3, 0, 0))


@pytest.mark.parametrize('missing', range(1, 9))
def test_rejects_truncated_final_tensor_offset(missing):
    with pytest.raises(EOFError):
        parse_gguf_header(io.BytesIO(tensor_header()[:-missing]))


def test_rejects_truncated_final_metadata_string():
    payload = (b'GGUF' + struct.pack('<IQQ', 3, 0, 1)
               + struct.pack('<Q', 3) + b'key' + struct.pack('<IQ', 8, 5) + b'ab')
    with pytest.raises(EOFError):
        parse_gguf_header(io.BytesIO(payload))


class ShortReads(io.BytesIO):
    def read(self, size=-1):
        return super().read(min(size, 2))


def test_stream_can_return_partial_reads():
    assert parse_gguf_header(ShortReads(tensor_header()))['weight'] == {
        'shape': [3], 'dtype': 'F32'}
