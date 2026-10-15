"""GPT2Config serializes n_layer/n_head/n_embd rather than generic aliases."""
from modelinfo.architecture import extract_architecture
from modelinfo.calculator import calculate_footprint


def test_gpt2_config_provides_exact_architecture_without_tensor_scan():
    config = {"model_type": "gpt2", "n_layer": 12, "n_head": 12, "n_embd": 768}
    assert extract_architecture({}, config) == (12, 768, False)
    result = calculate_footprint({}, context_length=1024, config=config)
    assert result["kv_cache_bytes"] == 4 * 12 * 768 * 1024


def test_generic_config_fields_take_precedence_over_gpt2_aliases():
    config = {"n_layer": 12, "n_head": 12, "n_embd": 768,
              "num_hidden_layers": 2, "num_attention_heads": 4, "hidden_size": 256}
    assert extract_architecture({}, config) == (2, 256, False)
