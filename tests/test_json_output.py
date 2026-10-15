import json

import pytest

from modelinfo import cli


@pytest.fixture
def analyzed(monkeypatch):
    info = {
        "format_name": "GGUF",
        "footprint": {"total_params": 9007199254740993},
        "tensors": {},
    }
    monkeypatch.setattr(cli, "analyze_model", lambda *a, **kw: info)
    monkeypatch.setattr(
        cli,
        "print_model_info",
        lambda **kw: pytest.fail("Unexpected terminal renderer"),
    )
    monkeypatch.setattr(
        cli,
        "print_compare_info",
        lambda *a, **kw: pytest.fail("Unexpected comparison renderer"),
    )
    return info


def test_json_flag_defaults_to_false():
    assert cli.parse_args(["model.gguf"]).json is False


def test_json_single_model_is_machine_readable(analyzed, capsys):
    assert cli.main(["model.gguf", "--json"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == analyzed
    assert captured.err == ""
    assert "\x1b" not in captured.out


def test_json_comparison_preserves_input_order_and_names(analyzed, capsys):
    assert cli.main(["first/model.gguf", "second/other.gguf", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == [
        {"name": "model.gguf", "info": analyzed},
        {"name": "other.gguf", "info": analyzed},
    ]


def test_json_single_model_supports_vllm_metadata(analyzed, monkeypatch, capsys):
    from modelinfo import hardware

    monkeypatch.setattr(hardware, "resolve_gpu", lambda target: ("Test GPU", 24.0, 1))
    analyzed["footprint"]["vllm_metrics"] = {"max_serving_capacity": 1234}
    assert cli.main(["model.gguf", "--json", "--vllm"]) == 0
    assert (
        json.loads(capsys.readouterr().out)["footprint"]["vllm_metrics"][
            "max_serving_capacity"
        ]
        == 1234
    )
