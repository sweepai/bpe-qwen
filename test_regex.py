#!/usr/bin/env python3
"""Test what the regex pattern is matching."""

import regex as re

# The Qwen pre-tokenization pattern
pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+"

# Test cases from the failing tests
test_cases = [
    "def hello_world():\n    print('Hello, World!')",
    "   leading spaces",
    " mid  spaces ",
    "word1  word2   word3    word4",
    "    indented\n  less",
]

regex_obj = re.compile(pattern)

for test in test_cases:
    print(f"\nTest: {repr(test)}")
    print("Matches:")
    for match in regex_obj.finditer(test):
        print(f"  {repr(match.group())}")