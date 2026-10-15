from modelinfo.architecture import extract_architecture


def test_gguf_omitted_kv_heads_defaults_to_attention_heads():
    metadata = {'general.architecture': 'llama', 'llama.block_count': 32,
                'llama.embedding_length': 4096, 'llama.attention.head_count': 32}
    assert extract_architecture({'__metadata__': metadata}) == (32, 4096, False)


def test_gguf_explicit_grouped_query_heads_are_preserved():
    metadata = {'general.architecture': 'llama', 'llama.block_count': 32,
                'llama.embedding_length': 4096, 'llama.attention.head_count': 32,
                'llama.attention.head_count_kv': 8}
    assert extract_architecture({'__metadata__': metadata}) == (32, 1024, False)
