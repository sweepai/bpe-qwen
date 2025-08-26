#!/usr/bin/env python3
"""
Benchmark script for evaluating bpe-qwen tokenizer on WikiText dataset.
Compares performance against HuggingFace tokenizers.
"""

import time
import statistics
import json
from typing import List, Tuple
import bpe_qwen

def install_dependencies():
    """Install required dependencies if not available."""
    imports_needed = []
    
    try:
        import transformers
    except ImportError:
        imports_needed.append("transformers")
    
    try:
        import datasets
    except ImportError:
        imports_needed.append("datasets")
    
    if imports_needed:
        print(f"Installing dependencies: {', '.join(imports_needed)}")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + imports_needed)
        print("Dependencies installed successfully!\n")
    
    import transformers
    import datasets
    return transformers, datasets

def load_wikitext_dataset(dataset_name: str = "wikitext-103-raw-v1", split: str = "test", max_samples: int = None):
    """Load WikiText dataset."""
    _, datasets = install_dependencies()
    
    print(f"Loading {dataset_name} dataset ({split} split)...")
    
    # Load dataset
    dataset = datasets.load_dataset("wikitext", dataset_name, split=split)
    
    # Filter out empty texts
    texts = [text for text in dataset["text"] if text.strip()]
    
    if max_samples:
        texts = texts[:max_samples]
    
    print(f"Loaded {len(texts)} non-empty text samples")
    
    # Calculate dataset statistics
    total_chars = sum(len(text) for text in texts)
    avg_length = total_chars / len(texts) if texts else 0
    
    print(f"Total characters: {total_chars:,}")
    print(f"Average text length: {avg_length:.1f} chars")
    
    return texts

def benchmark_tokenizer_on_texts(tokenizer, texts: List[str], name: str, is_huggingface: bool = False) -> dict:
    """Benchmark a tokenizer on a list of texts."""
    print(f"\n{'='*60}")
    print(f"Benchmarking {name}")
    print(f"{'='*60}")
    
    results = {
        "name": name,
        "total_texts": len(texts),
        "encode_times": [],
        "decode_times": [],
        "token_counts": [],
        "total_chars": sum(len(text) for text in texts),
        "total_tokens": 0,
    }
    
    # Warmup
    print("Warming up tokenizer...")
    warmup_text = texts[0] if texts else "Hello world"
    for _ in range(10):
        if is_huggingface:
            _ = tokenizer(warmup_text, return_tensors=None)['input_ids']
        else:
            _ = tokenizer.encode(warmup_text)
    
    # Benchmark encoding
    print("Benchmarking encoding...")
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        if i % 1000 == 0:
            print(f"  Processing texts {i}/{len(texts)}...")
        
        for text in batch:
            # Measure encoding time
            start_time = time.perf_counter()
            if is_huggingface:
                tokens = tokenizer(text, return_tensors=None)['input_ids']
            else:
                tokens = tokenizer.encode(text)
            encode_time = time.perf_counter() - start_time
            
            results["encode_times"].append(encode_time)
            results["token_counts"].append(len(tokens))
            results["total_tokens"] += len(tokens)
            
            # Measure decoding time
            start_time = time.perf_counter()
            _ = tokenizer.decode(tokens)
            decode_time = time.perf_counter() - start_time
            
            results["decode_times"].append(decode_time)
    
    return results

def print_results(results: dict):
    """Print benchmark results."""
    name = results["name"]
    
    # Calculate statistics
    avg_encode_time = statistics.mean(results["encode_times"]) * 1000  # Convert to ms
    median_encode_time = statistics.median(results["encode_times"]) * 1000
    p95_encode_time = statistics.quantiles(results["encode_times"], n=20)[18] * 1000  # 95th percentile
    
    avg_decode_time = statistics.mean(results["decode_times"]) * 1000
    median_decode_time = statistics.median(results["decode_times"]) * 1000
    p95_decode_time = statistics.quantiles(results["decode_times"], n=20)[18] * 1000
    
    total_encode_time = sum(results["encode_times"])
    total_decode_time = sum(results["decode_times"])
    
    tokens_per_sec_encode = results["total_tokens"] / total_encode_time if total_encode_time > 0 else 0
    tokens_per_sec_decode = results["total_tokens"] / total_decode_time if total_decode_time > 0 else 0
    
    chars_per_sec_encode = results["total_chars"] / total_encode_time if total_encode_time > 0 else 0
    chars_per_sec_decode = results["total_chars"] / total_decode_time if total_decode_time > 0 else 0
    
    print(f"\n{name} Results:")
    print(f"  Total texts: {results['total_texts']:,}")
    print(f"  Total characters: {results['total_chars']:,}")
    print(f"  Total tokens: {results['total_tokens']:,}")
    print(f"  Compression ratio: {results['total_chars'] / results['total_tokens']:.2f} chars/token")
    
    print(f"\nEncoding Performance:")
    print(f"  Average: {avg_encode_time:.3f} ms/text")
    print(f"  Median: {median_encode_time:.3f} ms/text")
    print(f"  P95: {p95_encode_time:.3f} ms/text")
    print(f"  Throughput: {tokens_per_sec_encode:,.0f} tokens/sec")
    print(f"  Throughput: {chars_per_sec_encode:,.0f} chars/sec")
    
    print(f"\nDecoding Performance:")
    print(f"  Average: {avg_decode_time:.3f} ms/text")
    print(f"  Median: {median_decode_time:.3f} ms/text")
    print(f"  P95: {p95_decode_time:.3f} ms/text")
    print(f"  Throughput: {tokens_per_sec_decode:,.0f} tokens/sec")
    print(f"  Throughput: {chars_per_sec_decode:,.0f} chars/sec")
    
    return {
        "tokens_per_sec_encode": tokens_per_sec_encode,
        "tokens_per_sec_decode": tokens_per_sec_decode,
        "avg_encode_ms": avg_encode_time,
        "avg_decode_ms": avg_decode_time,
    }

def compare_tokenizers(results_list: List[Tuple[dict, dict]]):
    """Compare results from multiple tokenizers."""
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    
    if len(results_list) < 2:
        return
    
    bpe_results, bpe_metrics = next(((r, m) for r, m in results_list if "bpe-qwen" in r["name"]), (None, None))
    hf_results, hf_metrics = next(((r, m) for r, m in results_list if "HuggingFace" in r["name"]), (None, None))
    
    if bpe_metrics and hf_metrics:
        encode_speedup = hf_metrics["avg_encode_ms"] / bpe_metrics["avg_encode_ms"]
        decode_speedup = hf_metrics["avg_decode_ms"] / bpe_metrics["avg_decode_ms"]
        throughput_encode_speedup = bpe_metrics["tokens_per_sec_encode"] / hf_metrics["tokens_per_sec_encode"]
        throughput_decode_speedup = bpe_metrics["tokens_per_sec_decode"] / hf_metrics["tokens_per_sec_decode"]
        
        print(f"\nbpe-qwen vs HuggingFace:")
        print(f"  Encoding: {encode_speedup:.2f}x faster (avg latency)")
        print(f"  Decoding: {decode_speedup:.2f}x faster (avg latency)")
        print(f"  Encoding throughput: {throughput_encode_speedup:.2f}x higher")
        print(f"  Decoding throughput: {throughput_decode_speedup:.2f}x higher")
        
        # Token count comparison
        if bpe_results and hf_results:
            bpe_tokens = bpe_results["total_tokens"]
            hf_tokens = hf_results["total_tokens"]
            token_diff_pct = abs(bpe_tokens - hf_tokens) / hf_tokens * 100
            
            print(f"\nToken Count Comparison:")
            print(f"  bpe-qwen: {bpe_tokens:,} tokens")
            print(f"  HuggingFace: {hf_tokens:,} tokens")
            print(f"  Difference: {token_diff_pct:.2f}%")

def main():
    """Main benchmark function."""
    print("WikiText Benchmark for bpe-qwen Tokenizer")
    print("="*80)
    
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark bpe-qwen on WikiText dataset")
    parser.add_argument("--dataset", default="wikitext-103-raw-v1", 
                       choices=["wikitext-2-raw-v1", "wikitext-103-raw-v1"],
                       help="WikiText dataset to use")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"],
                       help="Dataset split to use")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to process (None for all)")
    parser.add_argument("--skip-hf", action="store_true",
                       help="Skip HuggingFace tokenizer benchmark")
    args = parser.parse_args()
    
    # Load dataset
    texts = load_wikitext_dataset(args.dataset, args.split, args.max_samples)
    
    if not texts:
        print("No texts loaded!")
        return
    
    results_list = []
    
    try:
        # Benchmark our tokenizer
        print("\n" + "="*80)
        print("Initializing bpe-qwen tokenizer...")
        start = time.time()
        qwen_tokenizer = bpe_qwen.QwenTokenizer("../data")
        load_time = time.time() - start
        print(f"✓ bpe-qwen loaded in {load_time:.2f}s")
        print(f"  Vocab size: {qwen_tokenizer.vocab_size()}")
        
        results = benchmark_tokenizer_on_texts(qwen_tokenizer, texts, "bpe-qwen (Rust)")
        metrics = print_results(results)
        results_list.append((results, metrics))
        
    except Exception as e:
        print(f"Error benchmarking bpe-qwen: {e}")
        import traceback
        traceback.print_exc()
    
    if not args.skip_hf:
        try:
            # Benchmark HuggingFace tokenizer
            transformers, _ = install_dependencies()
            
            print("\n" + "="*80)
            print("Initializing HuggingFace tokenizer...")
            start = time.time()
            try:
                # Try to load from HuggingFace Hub
                hf_tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
            except:
                # Fall back to local file
                print("Loading from local tokenizer.json...")
                hf_tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
            load_time = time.time() - start
            print(f"✓ HuggingFace tokenizer loaded in {load_time:.2f}s")
            print(f"  Vocab size: {hf_tokenizer.vocab_size}")
            
            results = benchmark_tokenizer_on_texts(hf_tokenizer, texts, "HuggingFace (Python)", is_huggingface=True)
            metrics = print_results(results)
            results_list.append((results, metrics))
            
        except Exception as e:
            print(f"Error benchmarking HuggingFace tokenizer: {e}")
            import traceback
            traceback.print_exc()
    
    # Compare results
    if results_list:
        compare_tokenizers(results_list)
    
    # Save results to JSON
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"wikitext_benchmark_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "split": args.split,
            "max_samples": args.max_samples,
            "num_texts": len(texts),
            "results": [r for r, _ in results_list]
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    main()