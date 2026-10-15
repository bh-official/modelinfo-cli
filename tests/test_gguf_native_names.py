from modelinfo.architecture import extract_architecture


def test_native_gguf_key_projection_names():
    tensors = {
        "__metadata__": {"general.architecture": "llama"},
        "blk.0.attn_k.weight": {"shape": [4096, 1024]},
        "blk.1.attn_k.weight": {"shape": [4096, 1024]},
    }
    assert extract_architecture(tensors) == (2, 1024, False)


def test_native_gguf_fused_projection_names():
    tensors = {
        "__metadata__": {"general.architecture": "gpt2"},
        "blk.0.attn_qkv.weight": {"shape": [768, 2304]},
    }
    assert extract_architecture(tensors) == (1, 768, True)
