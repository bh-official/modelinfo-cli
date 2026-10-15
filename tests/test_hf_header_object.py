import json
import struct

import pytest

from modelinfo.parsers import huggingface as hf


@pytest.mark.parametrize('header', [None, [], [1], 'weights', 42, True])
def test_remote_safetensors_header_requires_object(monkeypatch, header):
    payload = json.dumps(header).encode()
    monkeypatch.setattr(hf, '_make_request', lambda *a, **kw: struct.pack('<Q', len(payload)) + payload)
    with pytest.raises(ValueError, match='JSON object'):
        hf._fetch_safetensors_header('owner/model', 'model.safetensors')


def test_invalid_header_shard_is_recorded_as_missing(monkeypatch):
    payload = b'[]'
    monkeypatch.setattr(hf, '_make_request', lambda *a, **kw: struct.pack('<Q', len(payload)) + payload)
    tensors, missing = hf._fetch_shards_concurrently('owner/model', ['model.safetensors'], 1)
    assert tensors == {}
    assert missing == 1


@pytest.mark.parametrize('header', [None, [], [1], 'weights', 42, True])
def test_local_safetensors_header_requires_object(tmp_path, header):
    from modelinfo.parsers.safetensors import _read_single_header

    payload = json.dumps(header).encode()
    path = tmp_path / 'model.safetensors'
    path.write_bytes(struct.pack('<Q', len(payload)) + payload)
    with pytest.raises(ValueError, match='JSON object'):
        _read_single_header(str(path))
