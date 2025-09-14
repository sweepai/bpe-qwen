# Import the compiled extension module
from . import bpe_qwen  # This is the compiled .so file
from .hf_patch import patch_transformers, unpatch_transformers, QwenTokenizerFast
from .auto_linear_tokenizer import AutoLinearTokenizer, QwenLinearTokenizer, get_tokenizer

# Re-export the QwenTokenizer from the compiled module
QwenTokenizer = bpe_qwen.QwenTokenizer

# Re-export pretokenization functions
pretokenize_slow = bpe_qwen.pretokenize_slow
pretokenize_fast = bpe_qwen.pretokenize_fast
pretokenize_fast_indices = bpe_qwen.pretokenize_fast_indices
indices_to_strings = bpe_qwen.indices_to_strings

__doc__ = getattr(bpe_qwen, '__doc__', '')
if hasattr(bpe_qwen, "__all__"):
    __all__ = bpe_qwen.__all__ + ["patch_transformers", "unpatch_transformers", "QwenTokenizerFast",
                                   "AutoLinearTokenizer", "QwenLinearTokenizer", "get_tokenizer",
                                   "pretokenize_slow", "pretokenize_fast", "pretokenize_fast_indices",
                                   "indices_to_strings", "pretokenize_automata"]
else:
    __all__ = ["QwenTokenizer", "patch_transformers", "unpatch_transformers", "QwenTokenizerFast",
               "AutoLinearTokenizer", "QwenLinearTokenizer", "get_tokenizer",
               "pretokenize_slow", "pretokenize_fast", "pretokenize_fast_indices",
               "indices_to_strings"]
