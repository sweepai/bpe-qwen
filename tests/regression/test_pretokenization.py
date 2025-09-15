#!/usr/bin/env python3
"""
Regression tests for pretokenization edge cases.

These tests ensure that various edge cases in pretokenization are handled correctly,
particularly around contractions, quoted words, and whitespace distribution.
"""

import sys
import os
import pytest

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bpe_qwen.bpe_qwen import pretokenize_slow, pretokenize_fast


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
            slow = pretokenize_slow(text)
            fast = pretokenize_fast(text)

            assert slow == expected, f"Slow tokenization failed for '{text}': {slow} != {expected}"
            assert fast == expected, f"Fast tokenization failed for '{text}': {fast} != {expected}"
            assert slow == fast, f"Slow and fast don't match for '{text}'"

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
            slow = pretokenize_slow(text)
            fast = pretokenize_fast(text)

            assert slow == expected_slow, f"Slow tokenization failed for '{text}': {slow} != {expected_slow}"
            assert fast == expected_fast, f"Fast tokenization failed for '{text}': {fast} != {expected_fast}"

    def test_verbose_in_context(self):
        """Test the specific 'verbose' edge case that was failing."""
        text = "                'verbose': True"
        expected = ['               ', " '", 'verbose', "':", ' True']

        slow = pretokenize_slow(text)
        fast = pretokenize_fast(text)

        assert slow == expected, f"Slow tokenization failed: {slow} != {expected}"
        assert fast == expected, f"Fast tokenization failed: {fast} != {expected}"

    def test_mixed_contractions_and_quotes(self):
        """Test text with both real contractions and quoted words."""
        test_cases = [
            ("you're 'wrong'", ["you", "'re", " '", "wrong", "'"]),
            ("I've seen 'verbose' code", ["I", "'ve", " seen", " '", "verbose", "'", " code"]),
            ("it's 'test' time", ["it", "'s", " '", "test", "'", " time"]),
        ]

        for text, expected in test_cases:
            slow = pretokenize_slow(text)
            fast = pretokenize_fast(text)

            assert slow == expected, f"Slow tokenization failed for '{text}': {slow} != {expected}"
            assert fast == expected, f"Fast tokenization failed for '{text}': {fast} != {expected}"


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
            slow = pretokenize_slow(text)
            fast = pretokenize_fast(text)

            assert slow == expected, f"Slow tokenization failed for '{text}': {slow} != {expected}"
            assert fast == expected, f"Fast tokenization failed for '{text}': {fast} != {expected}"

    def test_multiple_spaces(self):
        """Test handling of multiple consecutive spaces."""
        test_cases = [
            ("hello   world", ["hello", "  ", " world"]),
            ("a  b   c    d", ["a", " ", " b", "  ", " c", "   ", " d"]),
            ("   leading", ["  ", " leading"]),
            ("trailing   ", ["trailing", "   "]),
        ]

        for text, expected in test_cases:
            slow = pretokenize_slow(text)
            fast = pretokenize_fast(text)

            assert slow == expected, f"Slow tokenization failed for '{text}': {slow} != {expected}"
            assert fast == expected, f"Fast tokenization failed for '{text}': {fast} != {expected}"

    def test_spaces_with_numbers(self):
        """Test that spaces before numbers are handled differently than before letters."""
        test_cases = [
            ("    1234", ['   ', ' ', '1', '2', '3', '4']),
            ("test  123", ["test", " ", " ", "1", "2", "3"]),
            ("  42", [" ", " ", "4", "2"]),
        ]

        for text, expected in test_cases:
            slow = pretokenize_slow(text)
            fast = pretokenize_fast(text)

            assert slow == expected, f"Slow tokenization failed for '{text}': {slow} != {expected}"
            assert fast == expected, f"Fast tokenization failed for '{text}': {fast} != {expected}"

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
            slow = pretokenize_slow(text)
            fast = pretokenize_fast(text)

            assert slow == expected, f"Slow tokenization failed for '{text}': {slow} != {expected}"
            assert fast == expected, f"Fast tokenization failed for '{text}': {fast} != {expected}"


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
            slow = pretokenize_slow(text)
            fast = pretokenize_fast(text)

            assert slow == expected, f"Slow tokenization failed for '{text}': {slow} != {expected}"
            assert fast == expected, f"Fast tokenization failed for '{text}': {fast} != {expected}"

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
            slow = pretokenize_slow(text)
            fast = pretokenize_fast(text)

            assert slow == expected_slow, f"Slow tokenization failed for '{text}': {slow} != {expected_slow}"
            assert fast == expected_fast, f"Fast tokenization failed for '{text}': {fast} != {expected_fast}"



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

        slow = pretokenize_slow(code)
        fast = pretokenize_fast(code)

        # Just verify they match - exact token list would be very long
        assert slow == fast, "Slow and fast tokenization don't match for Python code"

        # Check that the patterns are in the output (they appear with preceding spaces)
        tokens_str = ' '.join(slow)
        assert "' verbose" in tokens_str or "' chunk" in tokens_str, "Quoted words should be in output"

    def test_json_like_data(self):
        """Test tokenization of JSON-like data structures."""
        json_text = """{"name": "test", "options": {"verbose": true, "debug": false}}"""

        slow = pretokenize_slow(json_text)
        fast = pretokenize_fast(json_text)

        assert slow == fast, "Slow and fast tokenization don't match for JSON"


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
                slow = pretokenize_slow(text)
                fast = pretokenize_fast(text)

                # Verify both complete without panicking
                assert isinstance(slow, list), f"Slow tokenization failed for '{text}'"
                assert isinstance(fast, list), f"Fast tokenization failed for '{text}'"

                # Results should be consistent (though may differ between slow/fast for some cases)
                print(f"Text: {text!r}")
                print(f"Slow: {slow}")
                print(f"Fast: {fast}")

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
                slow = pretokenize_slow(text)
                fast = pretokenize_fast(text)
                assert isinstance(slow, list), f"Slow tokenization failed for '{text}'"
                assert isinstance(fast, list), f"Fast tokenization failed for '{text}'"
            except Exception as e:
                pytest.fail(f"Tokenization panicked for edge case '{text}': {e}")


class TestTokenizationMismatchEdgeCases:
    """Test cases for specific tokenization mismatches found in real data."""

    def test_java_annotations(self):
        """Test Java annotation patterns that cause tokenization mismatches."""
        test_cases = [
            # Java annotations should stay together
            "@JsonProperty\n\tprivate",
            "@Override\n\tpublic",
            "@Component\n\tclass",
            "@Autowired\n\tprivate",
            # Annotations with parameters
            "@JsonProperty(\"name\")\n\tprivate",
            "@RequestMapping(\"/api\")\n\tpublic",
        ]

        for text in test_cases:
            slow = pretokenize_slow(text)
            fast = pretokenize_fast(text)

            # Both should produce the same result
            assert slow == fast, f"Tokenization mismatch for '{text}': slow={slow}, fast={fast}"

            # Verify annotation stays together (not split into @ + annotation)
            tokens_str = ''.join(slow)
            assert tokens_str == text, f"Tokens don't reconstruct original text for '{text}'"

    def test_import_statements(self):
        """Test import statement patterns that cause tokenization mismatches."""
        test_cases = [
            # Import statements with quoted strings - these SHOULD match but DON'T
            '"context"\n\t"encoding/json"',
            '"fmt"\n\t"strings"',
            '"encoding/json"\n\t"fmt"',
        ]

        for text in test_cases:
            slow = pretokenize_slow(text)
            fast = pretokenize_fast(text)

            # This should pass but FAILS - demonstrating the bug
            assert slow == fast, f"BUG: Tokenization mismatch for '{text}': slow={slow}, fast={fast}"

    def test_method_calls_and_chaining(self):
        """Test method call patterns that cause tokenization mismatches."""
        test_cases = [
            # Method calls that SHOULD match but DON'T
            ".stream()\n\t\t\t\t.filter",
            ".reverse();\n\n\thead",
            ".file\t\t=",
        ]

        for text in test_cases:
            slow = pretokenize_slow(text)
            fast = pretokenize_fast(text)

            # This should pass but FAILS - demonstrating the bug
            assert slow == fast, f"BUG: Tokenization mismatch for '{text}': slow={slow}, fast={fast}"

    def test_pointer_and_variable_patterns(self):
        """Test C/C++ pointer and variable patterns that cause mismatches."""
        test_cases = [
            # Pointer patterns
            "*xrc_domain",
            "*ptr_variable",
            "**double_pointer",
            # Variable with underscores
            "var_name_with_underscores",
            "CONSTANT_VALUE_NAME",
            # Mixed patterns
            "*variable_ptr->field",
            "&reference_var",
        ]

        for text in test_cases:
            slow = pretokenize_slow(text)
            fast = pretokenize_fast(text)

            assert slow == fast, f"Tokenization mismatch for '{text}': slow={slow}, fast={fast}"

            # Verify tokens reconstruct original
            tokens_str = ''.join(slow)
            assert tokens_str == text, f"Tokens don't reconstruct original text for '{text}'"

    def test_file_extensions_and_paths(self):
        """Test file extension and path patterns that cause mismatches."""
        test_cases = [
            # File extensions
            ".java",
            ".go",
            ".ts",
            ".cpp",
            ".h",
            # File paths
            "src/main/java/com/example/Class.java",
            "pkg/github/pullrequests.go",
            "include/ibvcore.h",
            # Relative paths
            "../../../parent/file.txt",
            "./current/directory/file.js",
        ]

        for text in test_cases:
            slow = pretokenize_slow(text)
            fast = pretokenize_fast(text)

            assert slow == fast, f"Tokenization mismatch for '{text}': slow={slow}, fast={fast}"

            # Verify tokens reconstruct original
            tokens_str = ''.join(slow)
            assert tokens_str == text, f"Tokens don't reconstruct original text for '{text}'"

    def test_complex_code_patterns(self):
        """Test complex code patterns that combine multiple edge cases."""
        test_cases = [
            # Java class with annotations - SHOULD match but DON'T
            '@JsonProperty\n\tprivate String name;\n\t@Override\n\tpublic String toString() {',
            # Go import block
            'import (\n\t"context"\n\t"encoding/json"\n\t"fmt"\n)',
            # C pointer operations
            '*xrc_domain->field = value;\n**ptr_ptr = &variable;',
            # Method chaining with file operations
            '.stream()\n\t\t.filter(file -> file.endsWith(".java"))\n\t\t.collect()',
        ]

        for text in test_cases:
            slow = pretokenize_slow(text)
            fast = pretokenize_fast(text)

            # This should pass but FAILS - demonstrating the bug
            assert slow == fast, f"BUG: Tokenization mismatch for complex pattern:\nText: {text!r}\nSlow: {slow}\nFast: {fast}"

    def test_tokenization_mismatch_summary(self):
        """Summary test that documents the key tokenization differences causing mismatches."""
        edge_cases = [
            # These patterns SHOULD tokenize the same but DON'T
            '\t@Override',
            '\t"context"',
            '\t.filter',
            '\thead',
            '\n\n\t@Test\n',
        ]

        for text in edge_cases:
            slow = pretokenize_slow(text)
            fast = pretokenize_fast(text)

            # This should pass but FAILS - demonstrating the bug
            assert slow == fast, f"BUG: Tokenization mismatch for '{text}': slow={slow}, fast={fast}"


class TestPerformanceRegression:
    """Ensure fixes don't regress performance too much."""

    def test_performance_ratio(self):
        """Test that fast implementation is actually faster than slow on large text."""
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

        # Time slow implementation
        start = time.perf_counter()
        for _ in range(10):
            slow_result = pretokenize_slow(large_text)
        slow_time = time.perf_counter() - start

        # Time fast implementation
        start = time.perf_counter()
        for _ in range(10):
            fast_result = pretokenize_fast(large_text)
        fast_time = time.perf_counter() - start

        # Verify correctness
        assert slow_result == fast_result, "Results don't match"

        # Verify performance (fast should be faster)
        speedup = slow_time / fast_time
        assert speedup > 1.0, f"Fast implementation is slower than slow! Speedup: {speedup:.2f}x"

        # Log the speedup for information
        print(f"\nPerformance: Fast is {speedup:.2f}x faster than slow")


if __name__ == "__main__":
    # Run with pytest or directly
    pytest.main([__file__, "-v"])