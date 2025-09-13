#!/usr/bin/env python3
"""Test different regex patterns to find one that works without lookahead."""

import regex as re

# Test different patterns without lookahead
patterns = [
    # Original with lookahead (for reference)
    (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+", "Original (with lookahead)"),

    # Without lookahead - current attempt
    (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+", "Without lookahead"),

    # Alternative: match spaces more carefully
    (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+| +(?=[^\s])| +$|\s+", "Alternative with end anchor"),

    # Another approach: be more specific about space patterns
    (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\s\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+| +", "Simpler space handling"),
]

test_cases = [
    "   leading spaces",
    " mid  spaces ",
    "word1  word2",
]

for pattern, name in patterns:
    print(f"\n{'='*60}")
    print(f"Pattern: {name}")
    print(f"{'='*60}")

    try:
        regex_obj = re.compile(pattern)

        for test in test_cases:
            print(f"\nTest: {repr(test)}")
            matches = []
            for match in regex_obj.finditer(test):
                matches.append(match.group())
                print(f"  {repr(match.group())}")

            # Check reconstruction
            reconstructed = ''.join(matches)
            if reconstructed != test:
                print(f"  ⚠️  Doesn't reconstruct: {repr(reconstructed)}")
    except Exception as e:
        print(f"Error compiling pattern: {e}")