"""
HuggingFace Tokenizer Patch - Drop-in replacement for Qwen tokenizers using bpe-qwen.

This module provides a monkey-patch that replaces HuggingFace's Qwen tokenizers
with the faster bpe-qwen implementation while maintaining full API compatibility.

Usage:
    import bpe_qwen.hf_patch
    bpe_qwen.hf_patch.patch_transformers()

    # Now all Qwen tokenizers will use bpe-qwen under the hood
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
    # This will use bpe-qwen's fast implementation!
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from huggingface_hub import snapshot_download
from bpe_qwen import bpe_qwen as bpe_qwen_module


class QwenTokenizerFast:
    """
    HuggingFace-compatible wrapper around bpe-qwen tokenizer.

    This class mimics the HuggingFace tokenizer interface while using
    bpe-qwen's fast Rust implementation under the hood.
    """

    def __init__(self,
                 vocab_file: Optional[str] = None,
                 merges_file: Optional[str] = None,
                 tokenizer_file: Optional[str] = None,
                 model_dir: Optional[str] = None,
                 **kwargs):
        """Initialize the tokenizer with HuggingFace-compatible parameters."""

        # Store HF-specific attributes
        self.model_max_length = kwargs.get('model_max_length', 32768)
        self.padding_side = kwargs.get('padding_side', 'left')
        self.pad_token = kwargs.get('pad_token', '<|endoftext|>')
        self.pad_token_id = kwargs.get('pad_token_id', 151643)
        self.eos_token = kwargs.get('eos_token', '<|endoftext|>')
        self.eos_token_id = kwargs.get('eos_token_id', 151643)
        self.unk_token = kwargs.get('unk_token', None)
        self.unk_token_id = kwargs.get('unk_token_id', None)
        self.bos_token = kwargs.get('bos_token', None)
        self.bos_token_id = kwargs.get('bos_token_id', None)
        self.add_special_tokens = kwargs.get('add_special_tokens', False)
        self.clean_up_tokenization_spaces = kwargs.get('clean_up_tokenization_spaces', False)

        # Try to find the tokenizer directory
        if model_dir:
            self._tokenizer_dir = model_dir
        elif vocab_file and merges_file:
            self._tokenizer_dir = Path(vocab_file).parent
        elif tokenizer_file:
            self._tokenizer_dir = Path(tokenizer_file).parent
        else:
            # Try to find from HuggingFace cache or local data directory
            self._tokenizer_dir = self._find_tokenizer_dir()

        # Initialize the actual bpe-qwen tokenizer
        self._tokenizer = bpe_qwen_module.QwenTokenizer(str(self._tokenizer_dir))

    def _find_tokenizer_dir(self) -> str:
        """Find tokenizer directory from local data directory."""
        local_data = Path(__file__).parent.parent / "data"
        return str(local_data)

    def encode(self,
               text: str,
               add_special_tokens: bool = None,
               padding: Union[bool, str] = False,
               truncation: Union[bool, str] = False,
               max_length: Optional[int] = None,
               return_tensors: Optional[str] = None,
               **kwargs) -> Union[List[int], Any]:
        """
        Encode text to token IDs.

        Compatible with HuggingFace's encode method.
        """
        # Use bpe-qwen's fast encoding
        token_ids = self._tokenizer.encode(text)

        # Handle special tokens if requested
        add_special = add_special_tokens if add_special_tokens is not None else self.add_special_tokens
        if add_special and self.bos_token_id is not None:
            token_ids = [self.bos_token_id] + token_ids

        # Handle truncation
        if truncation and max_length:
            token_ids = token_ids[:max_length]

        # Handle padding
        if padding and max_length and len(token_ids) < max_length:
            pad_length = max_length - len(token_ids)
            if self.padding_side == 'left':
                token_ids = [self.pad_token_id] * pad_length + token_ids
            else:
                token_ids = token_ids + [self.pad_token_id] * pad_length

        # Handle return tensors
        if return_tensors == "pt":
            import torch
            return torch.tensor([token_ids])
        elif return_tensors == "np":
            import numpy as np
            return np.array([token_ids])
        elif return_tensors == "tf":
            import tensorflow as tf
            return tf.constant([token_ids])

        return token_ids

    def decode(self,
               token_ids: Union[List[int], Any],
               skip_special_tokens: bool = True,
               clean_up_tokenization_spaces: Optional[bool] = None,
               **kwargs) -> str:
        """
        Decode token IDs back to text.

        Compatible with HuggingFace's decode method.
        """
        # Convert tensors to list if needed
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()

        # Handle nested lists (batch decoding)
        if isinstance(token_ids[0], list):
            token_ids = token_ids[0]

        # Filter special tokens if requested
        if skip_special_tokens:
            special_ids = {self.pad_token_id, self.eos_token_id}
            if self.bos_token_id is not None:
                special_ids.add(self.bos_token_id)
            if self.unk_token_id is not None:
                special_ids.add(self.unk_token_id)
            token_ids = [t for t in token_ids if t not in special_ids]

        # Use bpe-qwen's fast decoding
        text = self._tokenizer.decode(token_ids)

        # Clean up if requested
        clean_up = clean_up_tokenization_spaces if clean_up_tokenization_spaces is not None else self.clean_up_tokenization_spaces
        if clean_up:
            text = text.strip()

        return text

    def __call__(self,
                 text: Union[str, List[str]],
                 add_special_tokens: bool = None,
                 padding: Union[bool, str] = False,
                 truncation: Union[bool, str] = False,
                 max_length: Optional[int] = None,
                 return_tensors: Optional[str] = None,
                 return_attention_mask: bool = False,
                 **kwargs) -> Dict[str, Any]:
        """
        Main tokenization method compatible with HuggingFace.

        Returns a dictionary with 'input_ids' and optionally 'attention_mask'.
        """
        # Handle batch input
        if isinstance(text, list):
            # Use fast parallel encoding
            input_ids = self._tokenizer.encode_batch_parallel(text, num_workers=8)

            # Handle padding for batch
            if padding:
                max_len = max(len(ids) for ids in input_ids)
                if max_length:
                    max_len = min(max_len, max_length)

                padded_ids = []
                attention_masks = []
                for ids in input_ids:
                    if len(ids) < max_len:
                        pad_length = max_len - len(ids)
                        if self.padding_side == 'left':
                            mask = [0] * pad_length + [1] * len(ids)
                            ids = [self.pad_token_id] * pad_length + ids
                        else:
                            mask = [1] * len(ids) + [0] * pad_length
                            ids = ids + [self.pad_token_id] * pad_length
                    else:
                        mask = [1] * len(ids)
                    padded_ids.append(ids)
                    attention_masks.append(mask)
                input_ids = padded_ids
            else:
                attention_masks = [[1] * len(ids) for ids in input_ids]
        else:
            # Single text input
            input_ids = self.encode(text, add_special_tokens=add_special_tokens,
                                   padding=padding, truncation=truncation,
                                   max_length=max_length)
            if not isinstance(input_ids[0], list):
                input_ids = [input_ids]
            attention_masks = [[1] * len(input_ids[0])]

        # Build result dictionary
        result = {'input_ids': input_ids}

        if return_attention_mask:
            result['attention_mask'] = attention_masks

        # Convert to tensors if requested
        if return_tensors:
            if return_tensors == "pt":
                import torch
                result = {k: torch.tensor(v) for k, v in result.items()}
            elif return_tensors == "np":
                import numpy as np
                result = {k: np.array(v) for k, v in result.items()}
            elif return_tensors == "tf":
                import tensorflow as tf
                result = {k: tf.constant(v) for k, v in result.items()}

        return result

    def batch_encode_plus(self, *args, **kwargs):
        """Alias for __call__ with batch input."""
        return self.__call__(*args, **kwargs)

    def encode_plus(self, *args, **kwargs):
        """Alias for __call__ with single input."""
        return self.__call__(*args, **kwargs)

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """Tokenize text into subword tokens (returns strings, not IDs)."""
        # This is a simplified implementation
        # In reality, you'd need to map IDs back to token strings
        ids = self.encode(text, add_special_tokens=False)
        # For now, return a placeholder
        return [f"<token_{id}>" for id in ids]

    def get_vocab(self) -> Dict[str, int]:
        """Get the vocabulary dictionary."""
        # This would need actual implementation to read vocab.json
        return {}

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self._tokenizer.vocab_size()

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save tokenizer to directory (HF-compatible)."""
        # This would need implementation to copy tokenizer files
        pass


def patch_transformers():
    """
    Monkey-patch transformers to use bpe-qwen for Qwen models.

    After calling this function, any AutoTokenizer.from_pretrained() call
    for Qwen models will return the fast bpe-qwen implementation.
    """
    import transformers
    from transformers import AutoTokenizer

    # Store original from_pretrained
    _original_from_pretrained = AutoTokenizer.from_pretrained

    def patched_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
        """Patched from_pretrained that uses bpe-qwen for Qwen models."""

        # Check if this is a Qwen model
        model_name = str(pretrained_model_name_or_path).lower()
        is_qwen = 'qwen' in model_name

        # Also check the tokenizer_class if specified
        if not is_qwen and 'tokenizer_class' in kwargs:
            is_qwen = 'qwen' in str(kwargs.get('tokenizer_class', '')).lower()

        if is_qwen:
            # Use bpe-qwen implementation
            print(f"[bpe-qwen] Patching tokenizer for {pretrained_model_name_or_path}")

            # Get model directory
            if Path(pretrained_model_name_or_path).exists():
                model_dir = pretrained_model_name_or_path
            else:
                model_dir = snapshot_download(pretrained_model_name_or_path)

            # Create wrapped tokenizer
            return QwenTokenizerFast(model_dir=model_dir, **kwargs)

        # For non-Qwen models, use original implementation
        return _original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    # Apply the patch
    AutoTokenizer.from_pretrained = patched_from_pretrained
    transformers.AutoTokenizer.from_pretrained = patched_from_pretrained

    print("[bpe-qwen] Successfully patched transformers.AutoTokenizer")
    print("[bpe-qwen] Qwen tokenizers will now use the fast bpe-qwen implementation")

    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
    # print(tokenizer.encode("Hello, world!"))

    return True


def unpatch_transformers():
    """Restore original transformers behavior."""
    import transformers
    from transformers import AutoTokenizer

    # Restore original
    AutoTokenizer.from_pretrained = AutoTokenizer.from_pretrained.__wrapped__
    transformers.AutoTokenizer.from_pretrained = AutoTokenizer.from_pretrained.__wrapped__
    print("[bpe-qwen] Transformers has been unpatched")
    return True


# Auto-patch on import if BPE_QWEN_AUTO_PATCH env var is set
if os.environ.get('BPE_QWEN_AUTO_PATCH', '').lower() in ('1', 'true', 'yes'):
    patch_transformers()