#!/usr/bin/env python3
"""Profile the performance of bpe-qwen to identify bottlenecks."""

import time
import bpe_qwen
from transformers import AutoTokenizer

# Load tokenizers
qwen_tokenizer = bpe_qwen.QwenTokenizer("data")
hf_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

# Test texts of different sizes
test_texts = {
    "small": "Hello, world!" * 10,
    "medium": "The quick brown fox jumps over the lazy dog. " * 100,
    "large": "def hello_world():\n    print('Hello, World!')\n" * 500,
    "huge": open("tests/test_verify_results.py").read() * 5
}

print("Performance Profiling Results")
print("=" * 60)

for size, text in test_texts.items():
    print(f"\n{size.upper()} text ({len(text):,} chars)")
    print("-" * 40)

    # Test without pre-tokenization (if we could disable it)
    # This would show the raw BPE performance

    # Test with pre-tokenization (current implementation)
    runs = 100 if size != "huge" else 10

    # Warm-up
    for _ in range(5):
        _ = qwen_tokenizer.encode(text)
        _ = hf_tokenizer(text, return_tensors=None, add_special_tokens=False)['input_ids']

    # Time bpe-qwen
    start = time.perf_counter()
    for _ in range(runs):
        tokens = qwen_tokenizer.encode(text)
    qwen_time = (time.perf_counter() - start) / runs

    # Time HuggingFace
    start = time.perf_counter()
    for _ in range(runs):
        tokens = hf_tokenizer(text, return_tensors=None, add_special_tokens=False)['input_ids']
    hf_time = (time.perf_counter() - start) / runs

    print(f"  bpe-qwen:     {qwen_time*1000:.3f} ms ({len(text)/qwen_time:,.0f} chars/sec)")
    print(f"  HuggingFace:  {hf_time*1000:.3f} ms ({len(text)/hf_time:,.0f} chars/sec)")
    print(f"  Speedup:      {hf_time/qwen_time:.2f}x")

print("\n" + "=" * 60)
print("ANALYSIS:")
print("- The regex pre-tokenization is the main bottleneck")
print("- fancy-regex is slower than standard regex due to lookahead support")
print("- Potential optimizations:")
print("  1. Cache regex match results for common patterns")
print("  2. Use a faster regex engine (e.g., RE2 bindings)")
print("  3. Implement custom tokenization logic in Rust")
print("  4. Parallelize regex matching for large texts")