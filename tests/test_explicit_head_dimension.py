from modelinfo.architecture import extract_architecture
from modelinfo.calculator import calculate_footprint


def test_explicit_head_dimension_overrides_hidden_size_quotient():
    # Gemma2 2B configuration: 2304 / 8 is 288, but head_dim is 256.
    config = {"num_hidden_layers": 26, "num_attention_heads": 8,
              "num_key_value_heads": 4, "hidden_size": 2304, "head_dim": 256}
    assert extract_architecture({}, config) == (26, 1024, False)
    result = calculate_footprint({}, config=config, context_length=100)
    assert result["kv_cache_bytes"] == 4 * 26 * 1024 * 100


def test_config_without_explicit_dimension_retains_quotient():
    config = {"num_hidden_layers": 2, "num_attention_heads": 4,
              "hidden_size": 512}
    assert extract_architecture({}, config) == (2, 512, False)
