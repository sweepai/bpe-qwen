#!/usr/bin/env python3
"""Fair parallel benchmark comparing bpe-qwen against HuggingFace with proper parallelism control."""

import os
import time
import bpe_qwen
from transformers import AutoTokenizer
from datasets import load_dataset
import json
from datetime import datetime
import argparse

def benchmark_bpe_qwen_sequential(tokenizer, texts):
    """Benchmark bpe-qwen sequential encoding."""
    print("\n[bpe-qwen] Sequential (single-threaded)...")
    
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
    
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Speed: {tokens_per_sec/1e6:.2f}M tokens/sec")
    
    return elapsed, tokens_per_sec, results

def benchmark_bpe_qwen_parallel(tokenizer, texts, num_workers=8):
    """Benchmark bpe-qwen parallel batch encoding."""
    print(f"\n[bpe-qwen] Parallel ({num_workers} workers)...")
    
    # Warmup
    _ = tokenizer.encode_batch_parallel(texts[:10], num_workers)
    
    # Actual benchmark
    start = time.perf_counter()
    results = tokenizer.encode_batch_parallel(texts, num_workers)
    elapsed = time.perf_counter() - start
    
    total_tokens = sum(len(tokens) for tokens in results)
    tokens_per_sec = total_tokens / elapsed
    
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Speed: {tokens_per_sec/1e6:.2f}M tokens/sec")
    
    return elapsed, tokens_per_sec, results

def benchmark_huggingface_sequential(tokenizer, texts):
    """Benchmark HuggingFace with parallelism disabled."""
    # Disable HuggingFace's built-in parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print(f"\n[HuggingFace] Sequential (TOKENIZERS_PARALLELISM=false)...")
    
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
    
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Speed: {tokens_per_sec/1e6:.2f}M tokens/sec")
    
    return elapsed, tokens_per_sec, results

def benchmark_huggingface_parallel(tokenizer, texts):
    """Benchmark HuggingFace with its native parallelism enabled."""
    # Enable HuggingFace's built-in parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    print(f"\n[HuggingFace] Parallel (TOKENIZERS_PARALLELISM=true, native batch)...")
    
    # Warmup
    _ = tokenizer(texts[:10], padding=False, truncation=False, return_attention_mask=False)["input_ids"]
    
    # Actual benchmark - use native batch processing which leverages parallelism
    start = time.perf_counter()
    results = tokenizer(texts, padding=False, truncation=False, return_attention_mask=False)["input_ids"]
    elapsed = time.perf_counter() - start
    
    total_tokens = sum(len(tokens) for tokens in results)
    tokens_per_sec = total_tokens / elapsed
    
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Speed: {tokens_per_sec/1e6:.2f}M tokens/sec")
    
    return elapsed, tokens_per_sec, results

def main(num_texts=5000, num_workers=8):
    """Run fair parallel benchmarks with proper parallelism control."""
    print("="*80)
    print("Fair Parallel Tokenization Benchmark")
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
    print("\n" + "="*80)
    print("BENCHMARKING")
    print("="*80)
    
    results = {}
    
    # bpe-qwen sequential
    bpe_seq_time, bpe_seq_speed, bpe_seq_tokens = benchmark_bpe_qwen_sequential(bpe_tokenizer, texts)
    results["bpe_sequential"] = {"time": bpe_seq_time, "speed": bpe_seq_speed}
    
    # bpe-qwen parallel
    bpe_par_time, bpe_par_speed, bpe_par_tokens = benchmark_bpe_qwen_parallel(bpe_tokenizer, texts, num_workers)
    results["bpe_parallel"] = {"time": bpe_par_time, "speed": bpe_par_speed}
    
    # HuggingFace sequential (parallelism disabled)
    hf_seq_time, hf_seq_speed, hf_seq_tokens = benchmark_huggingface_sequential(hf_tokenizer, texts)
    results["hf_sequential"] = {"time": hf_seq_time, "speed": hf_seq_speed}
    
    # HuggingFace parallel (native parallelism enabled)
    hf_par_time, hf_par_speed, hf_par_tokens = benchmark_huggingface_parallel(hf_tokenizer, texts)
    results["hf_parallel"] = {"time": hf_par_time, "speed": hf_par_speed}
    
    # Results summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print("\n### Sequential Performance (Single-threaded):")
    print(f"bpe-qwen:      {bpe_seq_speed/1e6:.2f}M tokens/sec")
    print(f"HuggingFace:   {hf_seq_speed/1e6:.2f}M tokens/sec")
    print(f"Speedup:       {bpe_seq_speed/hf_seq_speed:.2f}x")
    
    print(f"\n### Parallel Performance:")
    print(f"bpe-qwen ({num_workers} workers):       {bpe_par_speed/1e6:.2f}M tokens/sec")
    print(f"HuggingFace (native parallel): {hf_par_speed/1e6:.2f}M tokens/sec")
    print(f"Speedup:                       {bpe_par_speed/hf_par_speed:.2f}x")
    
    print("\n### Internal Speedup from Parallelization:")
    print(f"bpe-qwen:      {bpe_par_speed/bpe_seq_speed:.2f}x (seq: {bpe_seq_speed/1e6:.2f}M → par: {bpe_par_speed/1e6:.2f}M)")
    print(f"HuggingFace:   {hf_par_speed/hf_seq_speed:.2f}x (seq: {hf_seq_speed/1e6:.2f}M → par: {hf_par_speed/1e6:.2f}M)")
    
    # Token consistency check
    print("\n### Token Consistency Check:")
    bpe_seq_total = sum(len(tokens) for tokens in bpe_seq_tokens)
    bpe_par_total = sum(len(tokens) for tokens in bpe_par_tokens)
    hf_seq_total = sum(len(tokens) for tokens in hf_seq_tokens)
    hf_par_total = sum(len(tokens) for tokens in hf_par_tokens)
    
    print(f"bpe-qwen sequential:    {bpe_seq_total:,} tokens")
    print(f"bpe-qwen parallel:      {bpe_par_total:,} tokens")
    print(f"HuggingFace sequential: {hf_seq_total:,} tokens")
    print(f"HuggingFace parallel:   {hf_par_total:,} tokens")
    
    # Verify correctness
    mismatches = 0
    for i in range(min(10, len(texts))):
        if bpe_seq_tokens[i] != bpe_par_tokens[i]:
            print(f"WARNING: Mismatch in bpe-qwen at index {i}")
            mismatches += 1
    
    if mismatches == 0:
        print("✓ bpe-qwen parallel results match sequential")
    
    # Save results
    import os
    os.makedirs("benchmark_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_results/fair_parallel_benchmark_{timestamp}.json"
    
    save_data = {
        "num_texts": len(texts),
        "num_workers": num_workers,
        "total_chars": total_chars,
        "results": results,
        "speedups": {
            "sequential": bpe_seq_speed / hf_seq_speed,
            "parallel": bpe_par_speed / hf_par_speed,
            "bpe_internal": bpe_par_speed / bpe_seq_speed,
            "hf_internal": hf_par_speed / hf_seq_speed
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nResults saved to {filename}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fair parallel tokenization benchmark")
    parser.add_argument("--num-texts", type=int, default=5000,
                       help="Number of texts to process (default: 5000)")
    parser.add_argument("--num-workers", type=int, default=8,
                       help="Number of parallel workers for bpe-qwen (default: 8)")
    
    args = parser.parse_args()
    
    main(args.num_texts, args.num_workers)