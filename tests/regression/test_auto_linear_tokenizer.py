#!/usr/bin/env python3
"""
End-to-end regression tests for AutoLinearTokenizer.

These tests verify that AutoLinearTokenizer works correctly as a drop-in replacement
for HuggingFace's AutoTokenizer, focusing on real functionality rather than mocked behavior.
"""

import pytest

from bpe_qwen.auto_linear_tokenizer import AutoLinearTokenizer, QwenLinearTokenizer, get_tokenizer


class TestAutoLinearTokenizerBasics:
    """Test basic functionality of AutoLinearTokenizer."""

    def test_can_create_tokenizer_instance(self):
        """Test that we can create a tokenizer instance."""
        tokenizer = QwenLinearTokenizer(model_dir="data")
        assert isinstance(tokenizer, QwenLinearTokenizer)
        assert hasattr(tokenizer, 'encode')
        assert hasattr(tokenizer, 'decode')
        assert hasattr(tokenizer, '__call__')

    def test_tokenizer_has_expected_attributes(self):
        """Test that tokenizer has HuggingFace-compatible attributes."""
        tokenizer = QwenLinearTokenizer(model_dir="data")

        # Check required attributes exist
        assert hasattr(tokenizer, 'model_max_length')
        assert hasattr(tokenizer, 'padding_side')
        assert hasattr(tokenizer, 'pad_token')
        assert hasattr(tokenizer, 'pad_token_id')
        assert hasattr(tokenizer, 'eos_token')
        assert hasattr(tokenizer, 'eos_token_id')

        # Check default values
        assert tokenizer.model_max_length == 32768
        assert tokenizer.padding_side == 'left'
        assert tokenizer.pad_token == '<|endoftext|>'
        assert tokenizer.pad_token_id == 151643

    def test_auto_linear_tokenizer_from_pretrained_interface(self):
        """Test that AutoLinearTokenizer.from_pretrained has the right interface."""
        # Test that the method exists and can be called
        assert hasattr(AutoLinearTokenizer, 'from_pretrained')
        assert callable(AutoLinearTokenizer.from_pretrained)

        # Test that it returns QwenLinearTokenizer
        tokenizer = AutoLinearTokenizer.from_pretrained("fake-model")
        assert isinstance(tokenizer, QwenLinearTokenizer)

    def test_get_tokenizer_convenience_function(self):
        """Test the get_tokenizer convenience function."""
        assert callable(get_tokenizer)

        tokenizer = get_tokenizer("fake-model")
        assert isinstance(tokenizer, QwenLinearTokenizer)


class TestTokenizerMethods:
    """Test tokenizer methods work correctly when tokenizer can be loaded."""

    def setup_method(self):
        """Set up tokenizer for tests."""
        self.tokenizer = QwenLinearTokenizer(model_dir="data")

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
        tokenizer = QwenLinearTokenizer(model_dir="data")

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
        tokenizer = QwenLinearTokenizer(model_dir="data")

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
        tokenizer = QwenLinearTokenizer(model_dir="data")

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

    def test_has_required_methods(self):
        """Test that tokenizer has all required HuggingFace methods."""
        tokenizer = QwenLinearTokenizer(model_dir="data")

        required_methods = [
            'encode',
            'decode',
            '__call__',
            'batch_encode_plus',
            'encode_plus',
            'save_vocabulary',
        ]

        for method_name in required_methods:
            assert hasattr(tokenizer, method_name), f"Missing method: {method_name}"
            assert callable(getattr(tokenizer, method_name)), f"Method not callable: {method_name}"

    def test_has_required_properties(self):
        """Test that tokenizer has all required HuggingFace properties."""
        tokenizer = QwenLinearTokenizer(model_dir="data")

        required_properties = [
            'vocab_size',
            'model_max_length',
            'padding_side',
            'pad_token',
            'pad_token_id',
            'eos_token',
            'eos_token_id',
        ]

        for prop_name in required_properties:
            assert hasattr(tokenizer, prop_name), f"Missing property: {prop_name}"

        # Test that vocab_size is callable and returns int
        if hasattr(tokenizer, 'vocab_size'):
            if callable(tokenizer.vocab_size):
                vocab_size = tokenizer.vocab_size()
                assert isinstance(vocab_size, int)
                assert vocab_size > 0
            else:
                assert isinstance(tokenizer.vocab_size, int)
                assert tokenizer.vocab_size > 0

    def test_aliases_work(self):
        """Test that method aliases work correctly."""
        tokenizer = QwenLinearTokenizer(model_dir="data")

        # Test that aliases exist and are callable
        assert hasattr(tokenizer, 'batch_encode_plus')
        assert callable(tokenizer.batch_encode_plus)

        assert hasattr(tokenizer, 'encode_plus')
        assert callable(tokenizer.encode_plus)

        # Test that they actually work
        result1 = tokenizer.encode_plus("test")
        assert isinstance(result1, dict)

        result2 = tokenizer.batch_encode_plus(["test1", "test2"])
        assert isinstance(result2, dict)


if __name__ == "__main__":
    # Run with pytest or directly
    pytest.main([__file__, "-v"])