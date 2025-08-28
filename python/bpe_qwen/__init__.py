from .bpe_qwen import *
from .hf_patch import patch_transformers, unpatch_transformers, QwenTokenizerFast

__doc__ = bpe_qwen.__doc__
if hasattr(bpe_qwen, "__all__"):
    __all__ = bpe_qwen.__all__ + ["patch_transformers", "unpatch_transformers", "QwenTokenizerFast"]
else:
    __all__ = ["patch_transformers", "unpatch_transformers", "QwenTokenizerFast"]
