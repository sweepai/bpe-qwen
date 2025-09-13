import sys
sys.path.append('.')
from bpe_qwen.bpe_qwen import pretokenize_slow, pretokenize_fast
import re

# Test the specific failing case
test_text = "            'chunk"

print("Test text:", repr(test_text))
print("\nCharacter analysis:")
for i, c in enumerate(test_text):
    print(f"  {i:2d}: {repr(c)} (whitespace: {c.isspace()}, alpha: {c.isalpha()}, numeric: {c.isnumeric()})")

print("\n\nSlow tokens:")
slow = pretokenize_slow(test_text)
for i, token in enumerate(slow):
    print(f"  {i}: {repr(token)}")

print("\nFast tokens:")
fast = pretokenize_fast(test_text)
for i, token in enumerate(fast):
    print(f"  {i}: {repr(token)}")

# Test with the standard regex pattern without lookahead
pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+"

# Compile with regex library (not re, as it doesn't support \p{L})
try:
    import regex
    regex_compiled = regex.compile(pattern)
    print("\n\nRegex matches (without lookahead):")
    matches = regex_compiled.findall(test_text)
    for i, match in enumerate(matches):
        print(f"  {i}: {repr(match)}")
except ImportError:
    print("\n\nCan't test regex pattern - regex module not installed")