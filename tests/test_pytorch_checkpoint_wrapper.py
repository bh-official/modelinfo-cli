import pickle
import zipfile

import pytest

from modelinfo.parsers.pytorch import parse_pytorch_header


@pytest.mark.parametrize('wrapper', ['state_dict', 'model_state_dict'])
def test_wrapped_state_dict_excludes_training_metadata(tmp_path, wrapper):
    weight = {'shape': [2, 3], 'dtype': 'F32'}
    data = {wrapper: {'layer.weight': weight}, 'epoch': 10, 'loss': 0.125}
    path = tmp_path / 'checkpoint.pt'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('checkpoint/data.pkl', pickle.dumps(data, protocol=2))
    assert parse_pytorch_header(str(path)) == {'layer.weight': weight}
