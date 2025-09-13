#!/usr/bin/env python3
"""Test how HuggingFace applies the pre-tokenization pattern."""

import regex as re
from transformers import AutoTokenizer

# Load HuggingFace tokenizer
hf_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

# The Qwen pre-tokenization pattern from HuggingFace
pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

# Test cases from the failing tests
test_cases = [
    "def hello_world():\n    print('Hello, World!')",
    "   leading spaces",
    " mid  spaces ",
    "word1  word2   word3    word4",
    "    indented\n  less",
]

print("HuggingFace pattern:", pattern)
regex_obj = re.compile(pattern)

for test in test_cases:
    print(f"\nTest: {repr(test)}")

    # What the regex matches
    print("Regex matches:")
    matches = []
    for match in regex_obj.finditer(test):
        matches.append(match.group())
        print(f"  {repr(match.group())}")

    # What HuggingFace tokenizer produces
    tokens = hf_tokenizer(test, return_tensors=None, add_special_tokens=False)['input_ids']
    decoded = [hf_tokenizer.decode([t], skip_special_tokens=False) for t in tokens]
    print("HuggingFace tokens:")
    for d in decoded:
        print(f"  {repr(d)}")

    # Check if they align
    reconstructed = ''.join(matches)
    if reconstructed != test:
        print(f"WARNING: Regex doesn't reconstruct original text!")
        print(f"  Original: {repr(test)}")
        print(f"  Reconstructed: {repr(reconstructed)}")