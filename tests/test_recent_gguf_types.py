import io
import struct

import pytest

from modelinfo.calculator import calculate_footprint
from modelinfo.parsers.gguf import parse_gguf_header


@pytest.mark.parametrize('type_id,dtype,block_size,block_bytes', [
    (34, 'TQ1_0', 256, 54), (35, 'TQ2_0', 256, 66), (39, 'MXFP4', 32, 17),
])
def test_gguf_ternary_and_mxfp4_memory(type_id, dtype, block_size, block_bytes):
    payload = (b'GGUF' + struct.pack('<IQQ', 3, 1, 0)
               + struct.pack('<Q', 6) + b'weight'
               + struct.pack('<IQIQ', 1, block_size, type_id, 0))
    tensors = parse_gguf_header(io.BytesIO(payload))
    assert tensors['weight']['dtype'] == dtype
    assert calculate_footprint(tensors)['base_memory_bytes'] == block_bytes
