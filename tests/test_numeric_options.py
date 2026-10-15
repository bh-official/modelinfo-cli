import pytest

from modelinfo.cli import parse_args


@pytest.mark.parametrize(
    "option,value",
    [
        ("--context", "0"),
        ("--context", "-1"),
        ("--max-vram", "0"),
        ("--max-vram", "-1"),
        ("--max-vram", "nan"),
        ("--max-vram", "inf"),
        ("--gpu-util", "0"),
        ("--gpu-util", "-0.1"),
        ("--gpu-util", "1.01"),
        ("--gpu-util", "nan"),
        ("--gpu-util", "inf"),
    ],
)
def test_invalid_numeric_options_fail_at_argument_parsing(option, value, capsys):
    with pytest.raises(SystemExit) as error:
        parse_args(["model.gguf", option, value])
    assert error.value.code == 2
    assert option in capsys.readouterr().err


@pytest.mark.parametrize("utilization", ["0.01", "0.9", "1"])
def test_valid_numeric_options_remain_supported(utilization):
    args = parse_args(
        ["model.gguf", "--context", "1", "--max-vram", "0.5", "--gpu-util", utilization]
    )
    assert args.context == 1
    assert args.max_vram == 0.5
    assert args.gpu_util == float(utilization)
