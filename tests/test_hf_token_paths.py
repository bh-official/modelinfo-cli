import pytest

from modelinfo.parsers.huggingface import _get_hf_token


@pytest.mark.parametrize('variable,relative', [
    ('HF_TOKEN_PATH', 'custom-token'),
    ('HF_HOME', 'custom-home/token'),
    ('XDG_CACHE_HOME', 'custom-cache/huggingface/token'),
])
def test_custom_huggingface_token_cache(monkeypatch, tmp_path, variable, relative):
    for name in ('HF_TOKEN', 'HF_TOKEN_PATH', 'HF_HOME', 'XDG_CACHE_HOME'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('USERPROFILE', str(tmp_path))
    token_file = tmp_path / relative
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text('fixture-token\n')
    value = token_file if variable == 'HF_TOKEN_PATH' else (token_file.parent if variable == 'HF_HOME' else token_file.parent.parent)
    monkeypatch.setenv(variable, str(value))
    assert _get_hf_token() == 'fixture-token'
    monkeypatch.setenv('HF_TOKEN', 'environment-fixture')
    assert _get_hf_token() == 'environment-fixture'
