#!/usr/bin/env python3
"""
Test to demonstrate the 'verbose' tokenization issue.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bpe_qwen.bpe_qwen import pretokenize_slow, pretokenize_fast

# Test sentence
text = "i like the word 'verbose'."

print("Input text:", repr(text))
print()

# Slow (fancy-regex with lookahead) - CORRECT
slow_result = pretokenize_slow(text)
print("Slow (fancy-regex) result:")
print(f"  Tokens: {slow_result}")
print(f"  Count: {len(slow_result)}")
print()

# Fast (two-pass) - INCORRECT
fast_result = pretokenize_fast(text)
print("Fast (two-pass) result:")
print(f"  Tokens: {fast_result}")
print(f"  Count: {len(fast_result)}")
print()

# Show the difference
print("The issue:")
print("- Slow correctly tokenizes 'verbose' as one token: \"'verbose\"")
print("- Fast incorrectly splits it as two tokens: \"'ve\" + \"rbose\"")
print()
print("Why this happens:")
print("1. The regex pattern starts with: (?i:'s|'t|'re|'ve|'m|'ll|'d)|...")
print("2. This pattern matches contractions like \"I've\", \"we're\", etc.")
print("3. When the regex sees 'verbose', it matches 've first (case-insensitive)")
print("4. So 'verbose' becomes 've + rbose")
print()
print("The slow version uses fancy-regex with lookahead which handles this correctly.")
print("The fast version uses standard regex (no lookahead) and can't easily fix this in the second pass.")