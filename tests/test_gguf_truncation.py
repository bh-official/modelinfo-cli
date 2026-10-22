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


def metadata_header(value_type, payload):
    return (b'GGUF' + struct.pack('<IQQ', 3, 0, 1)
            + struct.pack('<Q', 1) + b'k' + struct.pack('<I', value_type) + payload)


@pytest.mark.parametrize('payload', [
    struct.pack('<Q', 2**62) + b'abc',          # corrupt string length
    struct.pack('<IQ', 4, 2**62) + b'\0' * 16,  # corrupt numeric array length
])
def test_corrupt_lengths_hit_eof_instead_of_allocating(tmp_path, payload):
    # A real file matters: FileIO.read(n) preallocates n bytes, BytesIO does not.
    path = tmp_path / 'corrupt.gguf'
    path.write_bytes(metadata_header(8 if len(payload) == 11 else 9, payload))
    with pytest.raises(EOFError):
        parse_gguf_header(str(path))


def test_numeric_arrays_unpack_in_bulk():
    payload = struct.pack('<IQ3f', 6, 3, 1.5, -2.0, 0.25)
    assert parse_gguf_header(ShortReads(metadata_header(9, payload)))['__metadata__'] == {
        'k': [1.5, -2.0, 0.25]}
