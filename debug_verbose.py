import sys
sys.path.append('.')
from bpe_qwen.bpe_qwen import pretokenize_slow, pretokenize_fast

# Test the new failing case
test_text = "            'verbose': True"

print("Test text:", repr(test_text))

slow_tokens = pretokenize_slow(test_text)
fast_tokens = pretokenize_fast(test_text)

print("\nSlow tokens:")
for i, token in enumerate(slow_tokens):
    print(f"  {i:3d}: {repr(token)}")

print("\nFast tokens:")
for i, token in enumerate(fast_tokens):
    print(f"  {i:3d}: {repr(token)}")

print("\nDifferences:")
min_len = min(len(slow_tokens), len(fast_tokens))
for i in range(min_len):
    if slow_tokens[i] != fast_tokens[i]:
        print(f"  Position {i}: slow={repr(slow_tokens[i])} vs fast={repr(fast_tokens[i])}")

if len(slow_tokens) != len(fast_tokens):
    print(f"  Different lengths: slow has {len(slow_tokens)} tokens, fast has {len(fast_tokens)} tokens")