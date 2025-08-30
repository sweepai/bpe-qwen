# Import the compiled extension module
from . import bpe_qwen  # This is the compiled .so file
from .hf_patch import patch_transformers, unpatch_transformers, QwenTokenizerFast
from .auto_linear_tokenizer import AutoLinearTokenizer, QwenLinearTokenizer, get_tokenizer

# Re-export the QwenTokenizer from the compiled module
QwenTokenizer = bpe_qwen.QwenTokenizer

__doc__ = getattr(bpe_qwen, '__doc__', '')
if hasattr(bpe_qwen, "__all__"):
    __all__ = bpe_qwen.__all__ + ["patch_transformers", "unpatch_transformers", "QwenTokenizerFast",
                                   "AutoLinearTokenizer", "QwenLinearTokenizer", "get_tokenizer"]
else:
    __all__ = ["QwenTokenizer", "patch_transformers", "unpatch_transformers", "QwenTokenizerFast",
               "AutoLinearTokenizer", "QwenLinearTokenizer", "get_tokenizer"]
