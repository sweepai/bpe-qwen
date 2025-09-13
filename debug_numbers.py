#!/usr/bin/env python3
"""Debug script to understand number tokenization issues"""

from bpe_qwen import bpe_qwen

# Test cases that are failing
test_cases = [
    "123456789",  # Just numbers
    " 123456789",  # Space before numbers
    "  123456789",  # Multiple spaces before numbers
    "text 123456789",  # Text then space then numbers
    "'chunk_size': 1000",  # The problematic case from huge text
]

for text in test_cases:
    print(f"\nText: {repr(text)}")
    slow = bpe_qwen.pretokenize_slow(text)
    fast = bpe_qwen.pretokenize_fast(text)

    print(f"Slow: {slow}")
    print(f"Fast: {fast}")

    if slow != fast:
        print("MISMATCH!")
        for i in range(min(len(slow), len(fast))):
            if i < len(slow) and i < len(fast) and slow[i] != fast[i]:
                print(f"  Diff at {i}: {repr(slow[i])} vs {repr(fast[i])}")
                break