#!/usr/bin/env python3
"""
Regression tests for pretokenization_indices edge cases.

These tests ensure that the indices-based pretokenization algorithm matches
the behavior of the string-based pretokenization, particularly around
contractions, quoted words, and whitespace distribution.
"""

import sys
import os
import pytest

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bpe_qwen.bpe_qwen import pretokenize_slow, pretokenize_fast

# We'll need to import the Rust functions - assuming they're exposed in the Python module
try:
    from bpe_qwen.bpe_qwen import pretokenize_fast_indices, indices_to_strings
except ImportError:
    # If not available, we'll skip these tests
    pretokenize_fast_indices = None
    indices_to_strings = None


def skip_if_indices_not_available():
    """Skip test if indices functions are not available."""
    if pretokenize_fast_indices is None or indices_to_strings is None:
        pytest.skip("pretokenize_fast_indices or indices_to_strings not available")


def compare_with_string_tokenization(text):
    """Compare indices-based tokenization with string-based tokenization."""
    skip_if_indices_not_available()

    # Get string-based results
    slow_result = pretokenize_slow(text)
    fast_result = pretokenize_fast(text)

    # Get indices-based result
    indices = pretokenize_fast_indices(text)
    indices_result = indices_to_strings(text, indices)

    return slow_result, fast_result, indices_result


class TestContractionHandling:
    """Test cases for handling contractions and words that look like contractions."""

    def test_real_contractions(self):
        """Test that real contractions are tokenized correctly."""
        test_cases = [
            ("I've got it", ["I", "'ve", " got", " it"]),
            ("we're here", ["we", "'re", " here"]),
            ("they'll come", ["they", "'ll", " come"]),
            ("it's fine", ["it", "'s", " fine"]),
            ("I'd go", ["I", "'d", " go"]),
            ("can't stop", ["can", "'t", " stop"]),
            ("won't work", ["won", "'t", " work"]),
        ]

        for text, expected in test_cases:
            slow, fast, indices_result = compare_with_string_tokenization(text)

            assert slow == expected, f"Slow tokenization failed for '{text}': {slow} != {expected}"
            assert fast == expected, f"Fast tokenization failed for '{text}': {fast} != {expected}"
            assert indices_result == expected, f"Indices tokenization failed for '{text}': {indices_result} != {expected}"
            assert slow == fast == indices_result, f"Results don't match for '{text}'"

    def test_quoted_words_with_contraction_prefixes(self):
        """Test quoted words starting with contraction patterns.

        The slow (fancy-regex) version incorrectly splits these when they appear
        without preceding whitespace. The fast version fixes this issue.
        """
        test_cases = [
            # Without preceding space:
            # (text, expected_slow, expected_fast)
            ("'verbose'", ["'ve", "rbose", "'"], ["'verbose", "'"]),
            ("'test'", ["'t", "est", "'"], ["'test", "'"]),

            # With preceding space, both handle correctly
            (" 'verbose'", [" '", "verbose", "'"], [" '", "verbose", "'"]),
            (" 'test'", [" '", "test", "'"], [" '", "test", "'"]),
        ]

        for text, expected_slow, expected_fast in test_cases:
            slow, fast, indices_result = compare_with_string_tokenization(text)

            assert slow == expected_slow, f"Slow tokenization failed for '{text}': {slow} != {expected_slow}"
            assert fast == expected_fast, f"Fast tokenization failed for '{text}': {fast} != {expected_fast}"
            assert indices_result == expected_fast, f"Indices tokenization failed for '{text}': {indices_result} != {expected_fast}"

    def test_verbose_in_context(self):
        """Test the specific 'verbose' edge case that was failing."""
        text = "                'verbose': True"
        expected = ['               ', " '", 'verbose', "':", ' True']

        slow, fast, indices_result = compare_with_string_tokenization(text)

        assert slow == expected, f"Slow tokenization failed: {slow} != {expected}"
        assert fast == expected, f"Fast tokenization failed: {fast} != {expected}"
        assert indices_result == expected, f"Indices tokenization failed: {indices_result} != {expected}"

    def test_mixed_contractions_and_quotes(self):
        """Test text with both real contractions and quoted words."""
        test_cases = [
            ("you're 'wrong'", ["you", "'re", " '", "wrong", "'"]),
            ("I've seen 'verbose' code", ["I", "'ve", " seen", " '", "verbose", "'", " code"]),
            ("it's 'test' time", ["it", "'s", " '", "test", "'", " time"]),
        ]

        for text, expected in test_cases:
            slow, fast, indices_result = compare_with_string_tokenization(text)

            assert slow == expected, f"Slow tokenization failed for '{text}': {slow} != {expected}"
            assert fast == expected, f"Fast tokenization failed for '{text}': {fast} != {expected}"
            assert indices_result == expected, f"Indices tokenization failed for '{text}': {indices_result} != {expected}"


class TestWhitespaceDistribution:
    """Test cases for whitespace handling and distribution."""

    def test_spaces_before_quotes(self):
        """Test that spaces before quotes are handled correctly."""
        test_cases = [
            ("    'test': 123", ['   ', " '", 'test', "':", ' ', '1', '2', '3']),
            ("  'world'", [' ', " '", 'world', "'"]),
            (" 'hello'", [" '", 'hello', "'"]),
            ("                'chunk_size': 1000",
             ['               ', " '", 'chunk', '_size', "':", ' ', '1', '0', '0', '0']),
        ]

        for text, expected in test_cases:
            slow, fast, indices_result = compare_with_string_tokenization(text)

            assert slow == expected, f"Slow tokenization failed for '{text}': {slow} != {expected}"
            assert fast == expected, f"Fast tokenization failed for '{text}': {fast} != {expected}"
            assert indices_result == expected, f"Indices tokenization failed for '{text}': {indices_result} != {expected}"

    def test_multiple_spaces(self):
        """Test handling of multiple consecutive spaces."""
        test_cases = [
            ("hello   world", ["hello", "  ", " world"]),
            ("a  b   c    d", ["a", " ", " b", "  ", " c", "   ", " d"]),
            ("   leading", ["  ", " leading"]),
            ("trailing   ", ["trailing", "   "]),
        ]

        for text, expected in test_cases:
            slow, fast, indices_result = compare_with_string_tokenization(text)

            assert slow == expected, f"Slow tokenization failed for '{text}': {slow} != {expected}"
            assert fast == expected, f"Fast tokenization failed for '{text}': {fast} != {expected}"
            assert indices_result == expected, f"Indices tokenization failed for '{text}': {indices_result} != {expected}"

    def test_spaces_with_numbers(self):
        """Test that spaces before numbers are handled differently than before letters."""
        test_cases = [
            ("    1234", ['   ', ' ', '1', '2', '3', '4']),
            ("test  123", ["test", " ", " ", "1", "2", "3"]),
            ("  42", [" ", " ", "4", "2"]),
        ]

        for text, expected in test_cases:
            slow, fast, indices_result = compare_with_string_tokenization(text)

            assert slow == expected, f"Slow tokenization failed for '{text}': {slow} != {expected}"
            assert fast == expected, f"Fast tokenization failed for '{text}': {fast} != {expected}"
            assert indices_result == expected, f"Indices tokenization failed for '{text}': {indices_result} != {expected}"

    def test_newlines_and_special_whitespace(self):
        """Test handling of newlines and other special whitespace."""
        test_cases = [
            ("line1\nline2", ["line", "1", "\n", "line", "2"]),
            ("test\n\n\nmore", ["test", "\n\n\n", "more"]),
            ("line1\rline2", ["line", "1", "\r", "line", "2"]),
            ("line1\r\nline2", ["line", "1", "\r\n", "line", "2"]),
            # Tab followed by text starting with 'h' creates '\there' token
            ("tabs\t\there", ["tabs", "\t", "\there"]),
        ]

        for text, expected in test_cases:
            slow, fast, indices_result = compare_with_string_tokenization(text)

            assert slow == expected, f"Slow tokenization failed for '{text}': {slow} != {expected}"
            assert fast == expected, f"Fast tokenization failed for '{text}': {fast} != {expected}"
            assert indices_result == expected, f"Indices tokenization failed for '{text}': {indices_result} != {expected}"


class TestPunctuationAndSymbols:
    """Test cases for punctuation and symbol handling."""

    def test_punctuation_sequences(self):
        """Test handling of punctuation sequences."""
        test_cases = [
            ("Hello, world!", ["Hello", ",", " world", "!"]),
            ("Wait!!! Really???", ["Wait", "!!!", " Really", "???"]),
            ("test... okay", ["test", "...", " okay"]),
            # Punctuation followed by letter without space creates single token
            ("(a)", ["(a", ")"]),
            ("[test]", ["[test", "]"]),
            ("{code}", ["{code", "}"]),
        ]

        for text, expected in test_cases:
            slow, fast, indices_result = compare_with_string_tokenization(text)

            assert slow == expected, f"Slow tokenization failed for '{text}': {slow} != {expected}"
            assert fast == expected, f"Fast tokenization failed for '{text}': {fast} != {expected}"
            assert indices_result == expected, f"Indices tokenization failed for '{text}': {indices_result} != {expected}"

    def test_mixed_punctuation_and_quotes(self):
        """Test combinations of punctuation and quotes."""
        test_cases = [
            # Without space, slow splits 'test' but fast keeps it together
            ("'test': value", ["'t", "est", "':", " value"], ["'test", "':", " value"]),
            # Double quotes - slow splits the closing quote
            ('"quoted"', ['"quoted', '"'], ['"quoted"']),
            # With space before quote, both work correctly
            ("import 'module'", ["import", " '", "module", "'"], ["import", " '", "module", "'"]),
            ("{'key': 'value'}", ["{'", "key", "':", " '", "value", "'}"], ["{'", "key", "':", " '", "value", "'}"]),
        ]

        for text, expected_slow, expected_fast in test_cases:
            slow, fast, indices_result = compare_with_string_tokenization(text)

            assert slow == expected_slow, f"Slow tokenization failed for '{text}': {slow} != {expected_slow}"
            assert fast == expected_fast, f"Fast tokenization failed for '{text}': {fast} != {expected_fast}"
            assert indices_result == expected_fast, f"Indices tokenization failed for '{text}': {indices_result} != {expected_fast}"


class TestCodeSnippets:
    """Test pretokenization on code snippets."""

    def test_python_code(self):
        """Test tokenization of Python code snippets."""
        code = """def process_data(input_file, output_file, config=None):
    '''Process data from input file and write to output file.'''
    default_config = {
        'chunk_size': 1000,
        'verbose': True
    }"""

        slow, fast, indices_result = compare_with_string_tokenization(code)

        # Just verify they match - exact token list would be very long
        assert slow == fast, "Slow and fast tokenization don't match for Python code"
        assert indices_result == fast, "Indices and fast tokenization don't match for Python code"

        # Check that the patterns are in the output (they appear with preceding spaces)
        tokens_str = ' '.join(slow)
        assert "' verbose" in tokens_str or "' chunk" in tokens_str, "Quoted words should be in output"

    def test_json_like_data(self):
        """Test tokenization of JSON-like data structures."""
        json_text = """{"name": "test", "options": {"verbose": true, "debug": false}}"""

        slow, fast, indices_result = compare_with_string_tokenization(json_text)

        assert slow == fast, "Slow and fast tokenization don't match for JSON"
        assert indices_result == fast, "Indices and fast tokenization don't match for JSON"


class TestPanicRegression:
    """Test cases that previously caused panics in the Rust code."""

    def test_unicode_character_panic(self):
        """Test that Unicode characters don't cause panics in contraction handling.

        This test reproduces a panic that occurred when text.chars().nth(start_pos - 1)
        returned None due to byte vs character position mismatch with Unicode characters.
        """
        # Test cases with Unicode characters that can cause byte/char position mismatches
        test_cases = [
            # Unicode characters before contractions
            "café's delicious",
            "naïve's approach",
            "résumé's format",
            # Unicode in various positions
            "🚀's launch",
            "测试's test",
            "Москва's weather",
            # Edge case: Unicode at start with contraction
            "ñ's character",
            # Multiple Unicode chars
            "café naïve's test",
            # Unicode with quoted words that look like contractions
            "café 'verbose' test",
            # Empty or minimal cases that might trigger edge conditions
            "'s",
            "a's",
            # Cases with Unicode and whitespace
            "  café's  test  ",
            "\tcafé's\ttest",
        ]

        for text in test_cases:
            try:
                # These should not panic
                slow, fast, indices_result = compare_with_string_tokenization(text)

                # Verify both complete without panicking
                assert isinstance(slow, list), f"Slow tokenization failed for '{text}'"
                assert isinstance(fast, list), f"Fast tokenization failed for '{text}'"
                assert isinstance(indices_result, list), f"Indices tokenization failed for '{text}'"

                # Results should be consistent (though may differ between slow/fast for some cases)
                print(f"Text: {text!r}")
                print(f"Slow: {slow}")
                print(f"Fast: {fast}")
                print(f"Indices: {indices_result}")

            except Exception as e:
                pytest.fail(f"Tokenization panicked for text '{text}': {e}")

    def test_empty_and_edge_cases(self):
        """Test edge cases that might cause panics."""
        edge_cases = [
            "",  # Empty string
            "'",  # Single quote
            "''",  # Double quote
            "'s",  # Just contraction
            " 's",  # Space + contraction
            "\n's",  # Newline + contraction
            "\t's",  # Tab + contraction
        ]

        for text in edge_cases:
            try:
                slow, fast, indices_result = compare_with_string_tokenization(text)
                assert isinstance(slow, list), f"Slow tokenization failed for '{text}'"
                assert isinstance(fast, list), f"Fast tokenization failed for '{text}'"
                assert isinstance(indices_result, list), f"Indices tokenization failed for '{text}'"
            except Exception as e:
                pytest.fail(f"Tokenization panicked for edge case '{text}': {e}")


class TestIndicesSpecific:
    """Test cases specific to the indices-based implementation."""

    def test_indices_format(self):
        """Test that indices are returned in the correct format (end positions only)."""
        skip_if_indices_not_available()

        text = "Hello world"
        indices = pretokenize_fast_indices(text)

        # Should return end positions only
        assert isinstance(indices, list), "Indices should be a list"
        assert all(isinstance(i, int) for i in indices), "All indices should be integers"

        # Convert back to strings to verify correctness
        strings = indices_to_strings(text, indices)
        expected = ["Hello", " world"]
        assert strings == expected, f"Converted strings don't match: {strings} != {expected}"

    def test_indices_empty_text(self):
        """Test indices with empty text."""
        skip_if_indices_not_available()

        indices = pretokenize_fast_indices("")
        assert indices == [], "Empty text should return empty indices"

        strings = indices_to_strings("", indices)
        assert strings == [], "Empty indices should return empty strings"

    def test_indices_consistency_with_fast(self):
        """Test that indices-based tokenization is consistent with fast tokenization."""
        test_cases = [
            "Hello world",
            "I've got it",
            "'verbose' test",
            "    spaces",
            "line1\nline2",
            "Hello, world!",
            "café's test",
        ]

        for text in test_cases:
            skip_if_indices_not_available()

            fast_result = pretokenize_fast(text)
            indices = pretokenize_fast_indices(text)
            indices_result = indices_to_strings(text, indices)

            assert indices_result == fast_result, f"Indices result doesn't match fast result for '{text}': {indices_result} != {fast_result}"


class TestPerformanceRegression:
    """Ensure fixes don't regress performance too much."""

    def test_performance_ratio(self):
        """Test that indices implementation is faster than string-based on large text."""
        skip_if_indices_not_available()

        import time

        # Create a large text with various patterns
        large_text = """
        def process_data(input_file, output_file, config=None):
            '''Process data from input file and write to output file.'''
            default_config = {
                'chunk_size': 1000,
                'encoding': 'utf-8',
                'compression': 'gzip',
                'verbose': True
            }
            I've got what you're looking for.
            Multiple   spaces   here.
            Special chars: @#$%^&*()
        """ * 100  # Repeat to make it large

        # Time fast implementation
        start = time.perf_counter()
        for _ in range(10):
            fast_result = pretokenize_fast(large_text)
        fast_time = time.perf_counter() - start

        # Time indices implementation
        start = time.perf_counter()
        for _ in range(10):
            indices = pretokenize_fast_indices(large_text)
            indices_result = indices_to_strings(large_text, indices)
        indices_time = time.perf_counter() - start

        # Verify correctness
        assert indices_result == fast_result, "Results don't match"

        # Log the performance comparison
        speedup = fast_time / indices_time if indices_time > 0 else float('inf')
        print(f"\nPerformance: Indices is {speedup:.2f}x faster than fast string-based")

        # Indices should be at least as fast (allowing for some variance)
        assert speedup >= 0.8, f"Indices implementation is significantly slower! Speedup: {speedup:.2f}x"


if __name__ == "__main__":
    # Run with pytest or directly
    pytest.main([__file__, "-v"])