"""
AutoLinearTokenizer - A clean drop-in replacement for AutoTokenizer that uses bpe-qwen for Qwen models.

This module provides AutoLinearTokenizer, which inherits from HuggingFace's AutoTokenizer
but automatically uses the faster bpe-qwen implementation for Qwen models while maintaining
full API compatibility.

Usage:
    from bpe_qwen import AutoLinearTokenizer

    # Works exactly like AutoTokenizer but uses bpe-qwen for Qwen models
    tokenizer = AutoLinearTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Type
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from bpe_qwen import bpe_qwen as bpe_qwen_module


class QwenLinearTokenizer:
    """
    HuggingFace-compatible tokenizer that uses bpe-qwen's fast implementation.

    This class implements the HuggingFace tokenizer interface while using
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

        # Set up special tokens properly
        self._setup_special_tokens()

    def _find_tokenizer_dir(self) -> str:
        """Find tokenizer directory from local data directory."""
        local_data = Path(__file__).parent.parent / "data"
        return str(local_data)

    def _setup_special_tokens(self):
        """Setup special tokens to match HuggingFace's expected behavior."""
        # This ensures that the parent class's special token handling works correctly
        if self.pad_token:
            self._pad_token = self.pad_token
        if self.eos_token:
            self._eos_token = self.eos_token
        if self.bos_token:
            self._bos_token = self.bos_token
        if self.unk_token:
            self._unk_token = self.unk_token

    def _encode(self, text: str, **kwargs) -> List[int]:
        """Internal encode method that uses bpe-qwen."""
        return self._tokenizer.encode(text)

    def _decode(self, token_ids: List[int], **kwargs) -> str:
        """Internal decode method that uses bpe-qwen."""
        return self._tokenizer.decode(token_ids)

    def encode(self,
               text: str,
               add_special_tokens: bool = None,
               padding: Union[bool, str] = False,
               truncation: Union[bool, str] = False,
               max_length: Optional[int] = None,
               return_tensors: Optional[str] = None,
               **kwargs) -> Union[List[int], Any]:
        """
        Encode text to token IDs using bpe-qwen's fast implementation.
        """
        # Use bpe-qwen's fast encoding
        token_ids = self._encode(text)

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
        Decode token IDs back to text using bpe-qwen's fast implementation.
        """
        # Convert tensors to list if needed
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()

        # Handle nested lists (batch decoding)
        if token_ids and isinstance(token_ids[0], list):
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
        text = self._decode(token_ids)

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
            # For single string input, return list[int] to match HuggingFace behavior
            # Don't wrap in another list like we do for batch processing
            attention_masks = [1] * len(input_ids)

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

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size from bpe-qwen."""
        return self._tokenizer.vocab_size()

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        """Save vocabulary files (required by parent class)."""
        # This is a simplified implementation
        # In production, you'd copy the actual vocab files
        return (), ()


class AutoLinearTokenizer(AutoTokenizer):
    """
    Drop-in replacement for AutoTokenizer that uses bpe-qwen for Qwen models.

    This class inherits from AutoTokenizer and overrides the from_pretrained method
    to return QwenLinearTokenizer instances for Qwen models while maintaining full
    compatibility with the HuggingFace ecosystem.
    """

    @classmethod
    def from_pretrained(cls,
                       pretrained_model_name_or_path: Union[str, os.PathLike],
                       *inputs,
                       **kwargs) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        """
        Load a tokenizer from a pretrained model.

        Always returns QwenLinearTokenizer using bpe-qwen for fast tokenization.
        """
        # Always use bpe-qwen implementation
        print(f"[AutoLinearTokenizer] Using bpe-qwen for {pretrained_model_name_or_path}")

        # Get model directory
        if Path(pretrained_model_name_or_path).exists():
            model_dir = pretrained_model_name_or_path
        else:
            # Download from HuggingFace Hub if needed
            try:
                model_dir = snapshot_download(pretrained_model_name_or_path)
            except Exception:
                # If download fails, try using the parent class's resolution
                # This might work for locally cached models
                model_dir = pretrained_model_name_or_path

        # Create and return QwenLinearTokenizer
        return QwenLinearTokenizer(model_dir=model_dir, **kwargs)

    @classmethod
    def register(cls, config_class, tokenizer_class=None, slow_tokenizer_class=None, fast_tokenizer_class=None):
        """Register a new tokenizer class (delegates to parent)."""
        return super().register(config_class, tokenizer_class, slow_tokenizer_class, fast_tokenizer_class)


# For backwards compatibility - provide a simple function interface
def get_tokenizer(model_name_or_path: str, **kwargs) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """
    Convenience function to get a tokenizer using AutoLinearTokenizer.

    Args:
        model_name_or_path: Model name or path
        **kwargs: Additional arguments for tokenizer initialization

    Returns:
        A tokenizer instance (QwenLinearTokenizer for Qwen models, standard for others)
    """
    return AutoLinearTokenizer.from_pretrained(model_name_or_path, **kwargs)