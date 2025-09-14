#!/usr/bin/env python3
"""
Test the exact context where 'verbose' gets mis-tokenized.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bpe_qwen.bpe_qwen import pretokenize_slow, pretokenize_fast

# The exact context from the code where the issue happens
text = """                'verbose': True"""

print("Input text:", repr(text))
print()

# Slow (fancy-regex with lookahead)
slow_result = pretokenize_slow(text)
print("Slow (fancy-regex) result:")
for i, token in enumerate(slow_result):
    print(f"  [{i}]: {repr(token)}")
print(f"Count: {len(slow_result)}")
print()

# Fast (two-pass)
fast_result = pretokenize_fast(text)
print("Fast (two-pass) result:")
for i, token in enumerate(fast_result):
    print(f"  [{i}]: {repr(token)}")
print(f"Count: {len(fast_result)}")
print()

# Check if they match
if slow_result == fast_result:
    print("✓ Results match!")
else:
    print("✗ Results differ!")
    print()
    print("The issue occurs when 'verbose' appears after whitespace and a quote.")
    print("The regex pattern (?i:'ve) at the start of the alternation matches")
    print("'ve inside 'verbose' before the longer pattern can match the full word.")