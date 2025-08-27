#!/usr/bin/env python3
"""Benchmark parallel batch encoding performance against HuggingFace."""

import time
import bpe_qwen
from transformers import AutoTokenizer
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import json
from datetime import datetime
import argparse

def benchmark_bpe_qwen_parallel(tokenizer, texts, num_workers=8):
    """Benchmark bpe-qwen parallel batch encoding."""
    print(f"\nBenchmarking bpe-qwen with {num_workers} workers...")
    
    # Warmup
    _ = tokenizer.encode_batch_parallel(texts[:10], num_workers)
    
    # Actual benchmark
    start = time.perf_counter()
    results = tokenizer.encode_batch_parallel(texts, num_workers)
    elapsed = time.perf_counter() - start
    
    total_tokens = sum(len(tokens) for tokens in results)
    tokens_per_sec = total_tokens / elapsed
    texts_per_sec = len(texts) / elapsed
    
    return {
        "elapsed": elapsed,
        "total_tokens": total_tokens,
        "tokens_per_sec": tokens_per_sec,
        "texts_per_sec": texts_per_sec,
        "results": results
    }

def benchmark_bpe_qwen_sequential(tokenizer, texts):
    """Benchmark bpe-qwen sequential encoding for comparison."""
    print("\nBenchmarking bpe-qwen sequential...")
    
    # Warmup
    for text in texts[:10]:
        _ = tokenizer.encode(text)
    
    # Actual benchmark
    start = time.perf_counter()
    results = []
    for text in texts:
        tokens = tokenizer.encode(text)
        results.append(tokens)
    elapsed = time.perf_counter() - start
    
    total_tokens = sum(len(tokens) for tokens in results)
    tokens_per_sec = total_tokens / elapsed
    texts_per_sec = len(texts) / elapsed
    
    return {
        "elapsed": elapsed,
        "total_tokens": total_tokens,
        "tokens_per_sec": tokens_per_sec,
        "texts_per_sec": texts_per_sec,
        "results": results
    }

def benchmark_huggingface_parallel(tokenizer, texts, num_workers=8):
    """Benchmark HuggingFace with thread pool (simulating parallel)."""
    print(f"\nBenchmarking HuggingFace with {num_workers} threads...")
    
    # Warmup
    _ = tokenizer(texts[:10])["input_ids"]
    
    # Method 1: Batch processing (HF's native batching)
    print("  Method 1: Native batch processing...")
    start = time.perf_counter()
    results_batch = tokenizer(texts, padding=False, truncation=False, return_attention_mask=False)["input_ids"]
    batch_time = time.perf_counter() - start
    
    # Method 2: Thread pool (simulating parallel)
    def encode_single(text):
        return tokenizer.encode(text, add_special_tokens=False)
    
    print(f"  Method 2: ThreadPoolExecutor with {num_workers} workers...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        start = time.perf_counter()
        results_threads = list(executor.map(encode_single, texts))
        thread_time = time.perf_counter() - start
    
    # Use the faster method
    if batch_time < thread_time:
        print(f"  Using native batch (faster: {batch_time:.2f}s vs {thread_time:.2f}s)")
        elapsed = batch_time
        results = results_batch
    else:
        print(f"  Using thread pool (faster: {thread_time:.2f}s vs {batch_time:.2f}s)")
        elapsed = thread_time
        results = results_threads
    
    total_tokens = sum(len(tokens) for tokens in results)
    tokens_per_sec = total_tokens / elapsed
    texts_per_sec = len(texts) / elapsed
    
    return {
        "elapsed": elapsed,
        "total_tokens": total_tokens,
        "tokens_per_sec": tokens_per_sec,
        "texts_per_sec": texts_per_sec,
        "results": results
    }

def benchmark_huggingface_sequential(tokenizer, texts):
    """Benchmark HuggingFace sequential encoding for comparison."""
    print("\nBenchmarking HuggingFace sequential...")
    
    # Warmup
    for text in texts[:10]:
        _ = tokenizer.encode(text, add_special_tokens=False)
    
    # Actual benchmark
    start = time.perf_counter()
    results = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        results.append(tokens)
    elapsed = time.perf_counter() - start
    
    total_tokens = sum(len(tokens) for tokens in results)
    tokens_per_sec = total_tokens / elapsed
    texts_per_sec = len(texts) / elapsed
    
    return {
        "elapsed": elapsed,
        "total_tokens": total_tokens,
        "tokens_per_sec": tokens_per_sec,
        "texts_per_sec": texts_per_sec,
        "results": results
    }

def main(num_texts=5000, num_workers=8):
    """Run comprehensive parallel batch benchmarks."""
    print("="*80)
    print("Parallel Batch Tokenization Benchmark")
    print("="*80)
    
    # Load dataset
    print(f"\nLoading WikiText dataset ({num_texts} texts)...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test", trust_remote_code=True)
    texts = [item['text'] for item in dataset if item['text'].strip()][:num_texts]
    print(f"Loaded {len(texts)} non-empty texts")
    
    total_chars = sum(len(text) for text in texts)
    print(f"Total characters: {total_chars:,}")
    print(f"Average text length: {total_chars/len(texts):.1f} chars")
    
    # Initialize tokenizers
    print("\n" + "="*80)
    print("Initializing tokenizers...")
    
    # bpe-qwen
    bpe_tokenizer = bpe_qwen.QwenTokenizer("data/")
    print("✓ bpe-qwen tokenizer loaded")
    
    # HuggingFace
    hf_tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        trust_remote_code=True
    )
    print("✓ HuggingFace tokenizer loaded")
    
    # Run benchmarks
    results = {}
    
    print("\n" + "="*80)
    print("BENCHMARKING")
    print("="*80)
    
    # bpe-qwen sequential
    results["bpe_qwen_sequential"] = benchmark_bpe_qwen_sequential(bpe_tokenizer, texts)
    
    # bpe-qwen parallel
    results["bpe_qwen_parallel"] = benchmark_bpe_qwen_parallel(bpe_tokenizer, texts, num_workers)
    
    # HuggingFace sequential
    results["huggingface_sequential"] = benchmark_huggingface_sequential(hf_tokenizer, texts)
    
    # HuggingFace parallel/batch
    results["huggingface_parallel"] = benchmark_huggingface_parallel(hf_tokenizer, texts, num_workers)
    
    # Calculate speedups
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Sequential comparison
    print("\n### Sequential Performance:")
    print(f"bpe-qwen:      {results['bpe_qwen_sequential']['elapsed']:.2f}s " +
          f"({results['bpe_qwen_sequential']['tokens_per_sec']/1e6:.2f}M tokens/sec)")
    print(f"HuggingFace:   {results['huggingface_sequential']['elapsed']:.2f}s " +
          f"({results['huggingface_sequential']['tokens_per_sec']/1e6:.2f}M tokens/sec)")
    seq_speedup = results['huggingface_sequential']['elapsed'] / results['bpe_qwen_sequential']['elapsed']
    print(f"Sequential speedup: {seq_speedup:.2f}x")
    
    # Parallel comparison
    print(f"\n### Parallel Performance ({num_workers} workers):")
    print(f"bpe-qwen:      {results['bpe_qwen_parallel']['elapsed']:.2f}s " +
          f"({results['bpe_qwen_parallel']['tokens_per_sec']/1e6:.2f}M tokens/sec)")
    print(f"HuggingFace:   {results['huggingface_parallel']['elapsed']:.2f}s " +
          f"({results['huggingface_parallel']['tokens_per_sec']/1e6:.2f}M tokens/sec)")
    par_speedup = results['huggingface_parallel']['elapsed'] / results['bpe_qwen_parallel']['elapsed']
    print(f"Parallel speedup: {par_speedup:.2f}x")
    
    # Internal speedup (parallel vs sequential)
    print("\n### Parallelization Benefit:")
    bpe_internal = results['bpe_qwen_sequential']['elapsed'] / results['bpe_qwen_parallel']['elapsed']
    hf_internal = results['huggingface_sequential']['elapsed'] / results['huggingface_parallel']['elapsed']
    print(f"bpe-qwen parallel speedup:      {bpe_internal:.2f}x vs sequential")
    print(f"HuggingFace parallel speedup:   {hf_internal:.2f}x vs sequential")
    
    # Token consistency check
    print("\n### Token Consistency Check:")
    bpe_seq_tokens = results['bpe_qwen_sequential']['total_tokens']
    bpe_par_tokens = results['bpe_qwen_parallel']['total_tokens']
    hf_seq_tokens = results['huggingface_sequential']['total_tokens']
    hf_par_tokens = results['huggingface_parallel']['total_tokens']
    
    print(f"bpe-qwen sequential:  {bpe_seq_tokens:,} tokens")
    print(f"bpe-qwen parallel:    {bpe_par_tokens:,} tokens")
    print(f"HuggingFace sequential: {hf_seq_tokens:,} tokens")
    print(f"HuggingFace parallel:   {hf_par_tokens:,} tokens")
    
    # Verify correctness
    for i in range(min(10, len(texts))):
        if results['bpe_qwen_sequential']['results'][i] != results['bpe_qwen_parallel']['results'][i]:
            print(f"WARNING: Mismatch in bpe-qwen at index {i}")
            break
    else:
        print("✓ bpe-qwen parallel results match sequential")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"parallel_benchmark_{timestamp}.json"
    
    # Prepare data for JSON (exclude actual token results for space)
    save_data = {
        "num_texts": num_texts,
        "num_workers": num_workers,
        "total_chars": total_chars,
        "results": {
            k: {kk: vv for kk, vv in v.items() if kk != "results"}
            for k, v in results.items()
        },
        "speedups": {
            "sequential": seq_speedup,
            "parallel": par_speedup,
            "bpe_qwen_internal": bpe_internal,
            "huggingface_internal": hf_internal
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nResults saved to {filename}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark parallel batch tokenization")
    parser.add_argument("--num-texts", type=int, default=5000,
                       help="Number of texts to process (default: 5000)")
    parser.add_argument("--num-workers", type=int, default=8,
                       help="Number of parallel workers (default: 8)")
    
    args = parser.parse_args()
    
    main(args.num_texts, args.num_workers)