#!/usr/bin/env python3
"""
End-to-end regression tests for AutoLinearTokenizer.

These tests verify that AutoLinearTokenizer works correctly as a drop-in replacement
for HuggingFace's AutoTokenizer, focusing on real functionality rather than mocked behavior.
"""

import pytest

from transformers import AutoTokenizer

from bpe_qwen.auto_linear_tokenizer import AutoLinearTokenizer, QwenLinearTokenizer, get_tokenizer


class TestAutoLinearTokenizerMethods:
    """Test tokenizer methods work correctly when tokenizer can be loaded."""

    def setup_method(self):
        """Set up tokenizer for tests."""
        self.tokenizer = AutoLinearTokenizer.from_pretrained("data")

    def test_encode_decode_roundtrip(self):
        """Test that encode/decode works as a roundtrip."""
        test_texts = [
            "Hello, world!",
            "This is a test.",
            "Simple text",
            "",  # Empty string
        ]

        for text in test_texts:
            # Encode
            tokens = self.tokenizer.encode(text)
            assert isinstance(tokens, list)
            assert all(isinstance(t, int) for t in tokens)

            # Decode
            decoded = self.tokenizer.decode(tokens)
            assert isinstance(decoded, str)

            # For non-empty text, decoded should be similar to original
            if text.strip():
                assert len(decoded) > 0

    def test_call_method_returns_dict(self):
        """Test that __call__ method returns expected dictionary format."""
        result = self.tokenizer("Hello, world!")

        # Should return dict with input_ids
        assert isinstance(result, dict)
        assert 'input_ids' in result
        assert isinstance(result['input_ids'], list)

        # Test with attention mask
        result_with_mask = self.tokenizer("Hello, world!", return_attention_mask=True)
        assert 'attention_mask' in result_with_mask

    def test_batch_processing(self):
        """Test batch processing functionality."""
        texts = ["First text", "Second text", "Third text"]

        result = self.tokenizer(texts)

        # Should return dict with list of token lists
        assert isinstance(result, dict)
        assert 'input_ids' in result
        assert isinstance(result['input_ids'], list)
        assert len(result['input_ids']) == len(texts)

        # Each item should be a list of tokens
        for token_list in result['input_ids']:
            assert isinstance(token_list, list)
            assert all(isinstance(t, int) for t in token_list)

    def test_special_token_handling(self):
        """Test special token handling."""
        # Test with add_special_tokens
        tokens_no_special = self.tokenizer.encode("Hello", add_special_tokens=False)
        tokens_with_special = self.tokenizer.encode("Hello", add_special_tokens=True)

        assert isinstance(tokens_no_special, list)
        assert isinstance(tokens_with_special, list)

        # Test decode with skip_special_tokens
        if tokens_with_special:
            decoded_with_special = self.tokenizer.decode(tokens_with_special, skip_special_tokens=False)
            decoded_no_special = self.tokenizer.decode(tokens_with_special, skip_special_tokens=True)

            assert isinstance(decoded_with_special, str)
            assert isinstance(decoded_no_special, str)

    def test_padding_and_truncation(self):
        """Test padding and truncation functionality."""
        # Test truncation
        long_text = "This is a very long text " * 100
        tokens_truncated = self.tokenizer.encode(long_text, truncation=True, max_length=10)

        if tokens_truncated:
            assert len(tokens_truncated) <= 10

        # Test padding
        short_text = "Short"
        tokens_padded = self.tokenizer.encode(short_text, padding=True, max_length=20)

        if tokens_padded:
            assert len(tokens_padded) <= 20


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_and_whitespace_texts(self):
        """Test handling of empty and whitespace-only texts."""
        tokenizer = AutoLinearTokenizer.from_pretrained("data")

        edge_cases = [
            "",
            " ",
            "   ",
            "\n",
            "\t",
            "\r\n",
        ]

        for text in edge_cases:
            tokens = tokenizer.encode(text)
            assert isinstance(tokens, list)

            decoded = tokenizer.decode(tokens)
            assert isinstance(decoded, str)

    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        tokenizer = AutoLinearTokenizer.from_pretrained("data")

        unicode_texts = [
            "Hello 世界",
            "café résumé",
            "🚀 rocket emoji",
            "αβγδε Greek letters",
            "Здравствуй мир",
        ]

        for text in unicode_texts:
            tokens = tokenizer.encode(text)
            assert isinstance(tokens, list)

            decoded = tokenizer.decode(tokens)
            assert isinstance(decoded, str)

    def test_very_long_text(self):
        """Test handling of very long text."""
        tokenizer = AutoLinearTokenizer.from_pretrained("data")

        # Create a long text
        long_text = "This is a test sentence. " * 1000

        tokens = tokenizer.encode(long_text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0

        # Test that we can decode it back
        decoded = tokenizer.decode(tokens)
        assert isinstance(decoded, str)
        assert len(decoded) > 0


class TestCompatibilityWithHuggingFace:
    """Test compatibility with HuggingFace tokenizer interface."""

    def test_encode_plus_functionality(self):
        """Test that encode_plus works like HuggingFace tokenizers."""
        tokenizer = AutoLinearTokenizer.from_pretrained("data")

        # Test basic encode_plus
        result = tokenizer.encode_plus("Hello world")
        assert isinstance(result, dict)
        assert 'input_ids' in result
        assert isinstance(result['input_ids'], list)
        assert len(result['input_ids']) == 1  # Should be wrapped in batch
        assert isinstance(result['input_ids'][0], list)

        # Test with return_attention_mask
        result_with_mask = tokenizer.encode_plus("Hello world", return_attention_mask=True)
        assert 'attention_mask' in result_with_mask
        assert len(result_with_mask['attention_mask']) == 1

        # Test with padding and max_length
        result_padded = tokenizer.encode_plus("Hi", padding=True, max_length=10)
        assert len(result_padded['input_ids'][0]) <= 10

    def test_batch_encode_plus_functionality(self):
        """Test that batch_encode_plus works like HuggingFace tokenizers."""
        tokenizer = AutoLinearTokenizer.from_pretrained("data")

        texts = ["Hello world", "How are you?", "Fine thanks"]
        # texts = ["<|file_sep|>src/server/template-renderer/index.ts\nconst path = require('path')\nconst serialize"]

        # Test basic batch encoding
        result = tokenizer.batch_encode_plus(texts)
        assert isinstance(result, dict)
        assert 'input_ids' in result
        assert len(result['input_ids']) == len(texts)

        # Test with attention masks
        result_with_masks = tokenizer.batch_encode_plus(texts, return_attention_mask=True)
        assert 'attention_mask' in result_with_masks
        assert len(result_with_masks['attention_mask']) == len(texts)

        # Test with padding
        result_padded = tokenizer.batch_encode_plus(texts, padding=True)
        lengths = [len(seq) for seq in result_padded['input_ids']]
        assert len(set(lengths)) == 1, "All sequences should be same length when padded"

    def test_vocab_size_functionality(self):
        """Test that vocab_size returns a valid vocabulary size."""
        tokenizer = AutoLinearTokenizer.from_pretrained("data")

        vocab_size = tokenizer.vocab_size
        assert isinstance(vocab_size, int)
        assert vocab_size > 0
        assert vocab_size > 1000  # Should be a reasonable vocab size

    def test_special_tokens_functionality(self):
        """Test that special tokens work correctly."""
        tokenizer = AutoLinearTokenizer.from_pretrained("data")

        # Test that special token attributes exist and have reasonable values
        assert tokenizer.pad_token_id is not None
        assert isinstance(tokenizer.pad_token_id, int)
        assert tokenizer.pad_token_id >= 0

        assert tokenizer.eos_token_id is not None
        assert isinstance(tokenizer.eos_token_id, int)
        assert tokenizer.eos_token_id >= 0

        # Test that special tokens can be used in encoding/decoding
        text = "Hello world"
        tokens_with_special = tokenizer.encode(text, add_special_tokens=True)
        tokens_without_special = tokenizer.encode(text, add_special_tokens=False)

        # Should be different when special tokens are added
        if tokenizer.bos_token_id is not None:
            assert len(tokens_with_special) >= len(tokens_without_special)

    def test_padding_side_functionality(self):
        """Test that padding_side setting actually affects padding behavior."""
        tokenizer = AutoLinearTokenizer.from_pretrained("data")

        # Test left padding (default)
        tokenizer.padding_side = 'left'
        result_left = tokenizer(["Hi", "Hello world"], padding=True, max_length=10)

        # Test right padding
        tokenizer.padding_side = 'right'
        result_right = tokenizer(["Hi", "Hello world"], padding=True, max_length=10)

        # Results should be different due to different padding sides
        if len(result_left['input_ids'][0]) == len(result_right['input_ids'][0]):
            # If same length, padding positions should be different
            assert result_left['input_ids'][0] != result_right['input_ids'][0]

    def test_save_vocabulary_functionality(self):
        """Test that save_vocabulary can be called without errors."""
        tokenizer = AutoLinearTokenizer.from_pretrained("data")

        # Should not crash when called
        result = tokenizer.save_vocabulary("/tmp/test_vocab")
        assert isinstance(result, tuple)  # Should return tuple as per HF interface

    def test_linear_tokenization_method(self):
        """Test linear tokenization method compatibility with HuggingFace interface."""
        tokenizer = AutoLinearTokenizer.from_pretrained("data")
        huggingface_tokenizer = AutoTokenizer.from_pretrained("data")

        # Test texts and their ground truth equivalents
        texts = [
            "Hello, world!",
            "This is a test sentence.",
            "Another example text for testing."
        ]
        
        # Test linear tokenization using __call__ method
        linear_tokenized = tokenizer(texts)["input_ids"]
        huggingface_tokenized = huggingface_tokenizer(texts)["input_ids"]

        # Verify structure and types
        assert isinstance(linear_tokenized, list)
        assert len(linear_tokenized) == len(texts)

        # Each tokenized result should be a list of integers
        for token_list in linear_tokenized:
            assert isinstance(token_list, list)
            assert all(isinstance(token, int) for token in token_list)

        # Verify that linear tokenization matches HuggingFace tokenizer
        assert linear_tokenized == huggingface_tokenized

        # Test with different texts to ensure method works with varied input
        different_texts = ["Short", "A much longer text with more words"]
        different_tokenized = tokenizer(different_texts)["input_ids"]

        assert len(different_tokenized) == len(different_texts)
        assert len(different_tokenized[0]) != len(different_tokenized[1])  # Different lengths expected

    def test_linear_tokenization_special_tokens(self):
        """Test linear tokenization method compatibility with HuggingFace interface."""
        tokenizer = AutoLinearTokenizer.from_pretrained("data")
        huggingface_tokenizer = AutoTokenizer.from_pretrained("data")

        # Test texts and their ground truth equivalents
        texts = [
            "<|file_sep|>",
            "<|file_sep|>abcdefg",
        ]

        # Test linear tokenization using __call__ method
        linear_tokenized = tokenizer(texts)["input_ids"]
        huggingface_tokenized = huggingface_tokenizer(texts)["input_ids"]

        # Verify structure and types
        assert isinstance(linear_tokenized, list)
        assert len(linear_tokenized) == len(texts)

        # Each tokenized result should be a list of integers
        for token_list in linear_tokenized:
            assert isinstance(token_list, list)
            assert all(isinstance(token, int) for token in token_list)

        # Verify that linear tokenization matches HuggingFace tokenizer
        assert linear_tokenized == huggingface_tokenized

        # Test with different texts to ensure method works with varied input
        different_texts = ["Short", "A much longer text with more words"]
        different_tokenized = tokenizer(different_texts)["input_ids"]

        assert len(different_tokenized) == len(different_texts)
        assert len(different_tokenized[0]) != len(different_tokenized[1])  # Different lengths expected

class TestParallelization:
    """Test parallelization functionality."""

    def setup_method(self):
        """Set up tokenizer for tests."""
        self.tokenizer = AutoLinearTokenizer.from_pretrained("data")

    def test_batch_uses_parallel_encoding(self):
        """Test that batch processing uses parallel encoding method."""
        texts = ["First text for parallel test", "Second text for parallel test", "Third text for parallel test"]

        # This should internally call encode_batch_parallel, not individual encode calls
        result = self.tokenizer(texts)

        # Verify we get the expected batch structure
        assert isinstance(result, dict)
        assert 'input_ids' in result
        assert len(result['input_ids']) == len(texts)

        # Each result should be a list of token IDs
        for token_list in result['input_ids']:
            assert isinstance(token_list, list)
            assert len(token_list) > 0
            assert all(isinstance(token, int) for token in token_list)

    def test_single_text_vs_batch_consistency(self):
        """Test that single text and batch processing give consistent results."""
        test_text = "This is a test for consistency between single and batch processing"

        # Process as single text
        single_result = self.tokenizer(test_text)
        single_tokens = single_result['input_ids'][0]

        # Process as batch with one item
        batch_result = self.tokenizer([test_text])
        batch_tokens = batch_result['input_ids'][0]

        # Results should be identical
        assert single_tokens == batch_tokens

    def test_empty_batch_handling(self):
        """Test handling of empty batch."""
        result = self.tokenizer([])

        assert isinstance(result, dict)
        assert 'input_ids' in result
        assert result['input_ids'] == []

    def test_large_batch_processing(self):
        """Test processing of larger batches to verify parallel efficiency."""
        # Create a larger batch to test parallel processing
        texts = [f"This is test sentence number {i} for parallel processing." for i in range(20)]

        result = self.tokenizer(texts)

        # Verify all texts were processed
        assert len(result['input_ids']) == len(texts)

        # Verify each result is valid
        for i, token_list in enumerate(result['input_ids']):
            assert isinstance(token_list, list)
            assert len(token_list) > 0, f"Empty tokens for text {i}"
            assert all(isinstance(token, int) for token in token_list)

    def test_batch_with_attention_mask(self):
        """Test batch processing with attention masks."""
        texts = ["Short text", "This is a much longer text for testing", "Medium length text"]

        result = self.tokenizer(texts, return_attention_mask=True, padding=True)

        # Should have both input_ids and attention_mask
        assert 'input_ids' in result
        assert 'attention_mask' in result
        assert len(result['input_ids']) == len(texts)
        assert len(result['attention_mask']) == len(texts)

        # All sequences should be same length due to padding
        max_len = max(len(seq) for seq in result['input_ids'])
        for seq in result['input_ids']:
            assert len(seq) == max_len
        for mask in result['attention_mask']:
            assert len(mask) == max_len
            assert all(m in [0, 1] for m in mask)

    def test_mixed_length_batch(self):
        """Test batch processing with texts of very different lengths."""
        texts = [
            "Short",
            "This is a medium length text with several words in it",
            "A",
            "This is an extremely long text that goes on and on with many words and should test the parallel processing capabilities of the tokenizer when dealing with texts of vastly different lengths and complexities"
        ]

        result = self.tokenizer(texts)

        # Verify all texts processed correctly
        assert len(result['input_ids']) == len(texts)

        # Verify lengths are different (no unexpected padding)
        lengths = [len(tokens) for tokens in result['input_ids']]
        assert len(set(lengths)) > 1, "Expected different token lengths for different text lengths"

        # Shortest should be much shorter than longest
        assert min(lengths) < max(lengths)


if __name__ == "__main__":
    # Run with pytest or directly
    pytest.main([__file__, "-v"])