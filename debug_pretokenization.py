import sys
sys.path.append('.')
from bpe_qwen.bpe_qwen import pretokenize_fast, pretokenize_slow

# Test the failing case from HUGE text
test_text = """        default_config = {
            'chunk_size': 1000,"""

print("Test text:", repr(test_text))
print("\n")

slow_tokens = pretokenize_slow(test_text)
fast_tokens = pretokenize_fast(test_text)

print("Slow tokens:")
for i, token in enumerate(slow_tokens):
    print(f"  {i:3d}: {repr(token)}")

print("\nFast tokens:")
for i, token in enumerate(fast_tokens):
    print(f"  {i:3d}: {repr(token)}")

print("\nDifferences:")
for i, (s, f) in enumerate(zip(slow_tokens, fast_tokens)):
    if s != f:
        print(f"  Position {i}: slow={repr(s)} vs fast={repr(f)}")

# Test the failing case from MEDIUM text
test_text2 = """    1234"""

print("\n\nTest text 2:", repr(test_text2))
slow_tokens2 = pretokenize_slow(test_text2)
fast_tokens2 = pretokenize_fast(test_text2)

print("Slow tokens:")
for i, token in enumerate(slow_tokens2):
    print(f"  {i:3d}: {repr(token)}")

print("\nFast tokens:")
for i, token in enumerate(fast_tokens2):
    print(f"  {i:3d}: {repr(token)}")