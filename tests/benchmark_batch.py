#!/usr/bin/env python3
"""Benchmark batch encoding performance."""

import time
import bpe_qwen
from datasets import load_dataset

def benchmark_batch_encoding():
    # Initialize tokenizer
    tokenizer = bpe_qwen.QwenTokenizer("data/")
    
    # Load WikiText dataset
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test", trust_remote_code=True)
    texts = [item['text'] for item in dataset if item['text'].strip()][:1000]  # Use first 1000 texts
    
    print(f"Testing with {len(texts)} texts...")
    
    # Test individual encoding
    start = time.perf_counter()
    individual_results = []
    for text in texts:
        tokens = tokenizer.encode(text)
        individual_results.append(tokens)
    individual_time = time.perf_counter() - start
    
    # Test batch encoding
    start = time.perf_counter()
    batch_results = tokenizer.encode_batch(texts)
    batch_time = time.perf_counter() - start
    
    # Calculate speedup
    speedup = individual_time / batch_time
    
    print(f"\nResults:")
    print(f"Individual encoding: {individual_time:.3f}s")
    print(f"Batch encoding: {batch_time:.3f}s")
    print(f"Speedup: {speedup:.2f}x")
    
    # Verify results match
    for i, (ind, batch) in enumerate(zip(individual_results, batch_results)):
        if ind != batch:
            print(f"WARNING: Mismatch at index {i}")
            break
    else:
        print("✓ Results match!")
    
    return speedup

if __name__ == "__main__":
    speedup = benchmark_batch_encoding()
    
    # Return success if faster, failure if slower
    if speedup > 1.0:
        print(f"\n✅ Batch processing is {speedup:.2f}x faster!")
        exit(0)
    else:
        print(f"\n❌ Batch processing is slower ({speedup:.2f}x)")
        exit(1)