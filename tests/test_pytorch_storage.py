"""Fixed protocol-2 fixture using torch.save's storage persistent ID layout."""
import io
import pickle
import zipfile

import pytest

from modelinfo.parsers.pytorch import RestrictedUnpickler, parse_pytorch_header


@pytest.mark.parametrize('storage,dtype', [('FloatStorage', 'F32'), ('HalfStorage', 'F16'),
    ('BFloat16Storage', 'BF16'), ('IntStorage', 'I32'), ('LongStorage', 'I64'),
    ('DoubleStorage', 'F64'), ('ShortStorage', 'I16'), ('CharStorage', 'I8'),
    ('ByteStorage', 'U8'), ('BoolStorage', 'BOOL')])
def test_tensor_with_persistent_storage_id(tmp_path, storage, dtype):
    payload = (b'\x80\x02}X\x06\x00\x00\x00weightctorch._utils\n_rebuild_tensor_v2\n'
               b'((X\x07\x00\x00\x00storagectorch\n' + storage.encode() + b'\n'
               b'X\x01\x00\x00\x000X\x03\x00\x00\x00cpuK\x06tQK\x00K\x02K\x03\x86'
               b'K\x03K\x01\x86\x89ccollections\nOrderedDict\n)RtRs.')
    path = tmp_path / 'model.pt'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('model/data.pkl', payload)
    assert parse_pytorch_header(str(path)) == {'weight': {'shape': [2, 3], 'dtype': dtype}}


@pytest.mark.parametrize('pid', [None, (), ('module',), ('storage', int, '0', 'cpu', 6),
    ('storage', None, '0', 'cpu', 6)])
def test_invalid_persistent_ids_are_rejected(pid):
    unpickler = RestrictedUnpickler(io.BytesIO())
    with pytest.raises(pickle.UnpicklingError):
        unpickler.persistent_load(pid)
