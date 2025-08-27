# Make bpe_qwen a proper Python package
__version__ = "0.1.0"

# Import the main tokenizer
try:
    from .bpe_qwen import QwenTokenizer
except ImportError:
    # Try importing from the compiled module
    try:
        from bpe_qwen import QwenTokenizer
    except ImportError:
        pass

__all__ = ["QwenTokenizer"]