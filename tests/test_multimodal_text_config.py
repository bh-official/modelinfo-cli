from copy import deepcopy

from modelinfo.calculator import calculate_footprint


def test_lazy_multimodal_checkpoint_uses_text_backbone_kv_dimensions():
    config = {
        'model_type': 'llava',
        'vision_config': {'hidden_size': 1024, 'num_hidden_layers': 24},
        'text_config': {
            'hidden_size': 4096, 'num_hidden_layers': 32,
            'num_attention_heads': 32, 'num_key_value_heads': 8,
        },
    }
    original = deepcopy(config)
    tensors = {'__metadata__': {'lazy_fetch': True, 'total_size': 1024}}
    result = calculate_footprint(tensors, context_length=2048, config=config)
    assert result['num_layers'] == 32
    assert result['kv_dim'] == 1024
    assert result['kv_cache_bytes'] == 2 * 32 * 1024 * 2048 * 2
    assert config == original
