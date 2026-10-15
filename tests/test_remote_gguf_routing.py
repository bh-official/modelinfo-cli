import pytest

from modelinfo import cli
from modelinfo.parsers import huggingface


@pytest.mark.parametrize(
    "path", ["org/model/weights.gguf", "org/model/subdir/weights.GGUF"]
)
def test_explicit_hub_gguf_reaches_remote_parser(monkeypatch, path):
    monkeypatch.setattr(cli.os.path, "exists", lambda path: False)
    calls = []

    def fetch(repo, **kwargs):
        calls.append(repo)
        raise RuntimeError("remote parser reached")

    monkeypatch.setattr(huggingface, "fetch_huggingface_repo", fetch)
    with pytest.raises(RuntimeError, match="remote parser reached"):
        cli.analyze_model(path, context_override=128)
    assert calls == [path]


@pytest.mark.parametrize(
    "path",
    [
        "./org/model/weights.gguf",
        "/org/model/weights.gguf",
        "../org/model/weights.gguf",
    ],
)
def test_explicit_local_prefix_does_not_reach_hub(monkeypatch, path):
    monkeypatch.setattr(cli.os.path, "exists", lambda path: False)
    monkeypatch.setattr(
        huggingface,
        "fetch_huggingface_repo",
        lambda *a, **k: pytest.fail("Local path sent to Hub"),
    )
    with pytest.raises((OSError, ValueError)):
        cli.analyze_model(path, context_override=128)
