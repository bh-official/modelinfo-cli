import json
from unittest.mock import patch

from modelinfo.parsers.huggingface import _fetch_remote_safetensors_sharded


def test_remote_shards_include_only_tensors_assigned_by_index():
    index = {'weight_map': {'weight': 'one.safetensors'}, 'metadata': {'total_size': 8}}
    header = {'weight': {'shape': [2], 'dtype': 'F32'}, 'extra': {'shape': [100], 'dtype': 'F32'}}
    with patch('modelinfo.parsers.huggingface._make_request', return_value=json.dumps(index).encode()), patch('modelinfo.parsers.huggingface._fetch_safetensors_header', return_value=header):
        tensors, _ = _fetch_remote_safetensors_sharded('org/model', None, True, 10)
    assert set(tensors) == {'weight', '__metadata__'}


def test_remote_shard_missing_an_indexed_tensor_is_incomplete():
    index = {'weight_map': {'weight': 'one.safetensors'}}
    with patch('modelinfo.parsers.huggingface._make_request', return_value=json.dumps(index).encode()), patch('modelinfo.parsers.huggingface._fetch_safetensors_header', return_value={}):
        tensors, _ = _fetch_remote_safetensors_sharded('org/model', None, True, 10)
    assert tensors['__metadata__']['missing_shards'] == 1
