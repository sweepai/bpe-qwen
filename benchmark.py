#!/usr/bin/env python3
"""
Benchmark script comparing bpe-qwen against HuggingFace tokenizers.
"""

import time
import statistics
from typing import List, Tuple
import bpe_qwen

def install_dependencies():
    """Install required dependencies if not available."""
    try:
        import transformers
    except ImportError:
        print("Installing transformers...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
        import transformers
    
    return transformers

def create_test_texts() -> List[str]:
    """Create a variety of test texts for benchmarking."""
    return [
        # Short text
        "Hello, world!",
        
        # Medium text with code
        """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
        """,
        
        # Longer text with mixed content
        """
The quick brown fox jumps over the lazy dog. This pangram contains every letter of the alphabet.
Here's some code: `x = 42; y = x * 2; print(f"Result: {y}")`.
And some mathematical notation: ∑(i=1 to n) i = n(n+1)/2.
Unicode examples: 你好世界, Здравствуй мир, مرحبا بالعالم.
        """,
        
        # Very long text (repeated pattern)
        "This is a test sentence. " * 100,
        
        # Code-heavy text
        """
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x
        """,
    ]

def benchmark_tokenizer(tokenizer, texts: List[str], name: str) -> dict:
    """Benchmark a tokenizer on given texts."""
    print(f"\nBenchmarking {name}...")
    
    results = {
        "name": name,
        "encode_times": [],
        "decode_times": [],
        "token_counts": [],
        "accuracy": True,
        "total_tokens": 0
    }
    
    for i, text in enumerate(texts):
        print(f"  Text {i+1}/{len(texts)} ({len(text)} chars)")
        
        # Benchmark encoding
        start_time = time.perf_counter()
        if hasattr(tokenizer, 'encode'):
            tokens = tokenizer.encode(text)
        else:
            # HuggingFace tokenizer
            tokens = tokenizer(text, return_tensors=None)['input_ids']
        
        encode_time = time.perf_counter() - start_time
        results["encode_times"].append(encode_time)
        results["token_counts"].append(len(tokens))
        results["total_tokens"] += len(tokens)
        
        # Benchmark decoding
        start_time = time.perf_counter()
        if hasattr(tokenizer, 'decode'):
            decoded = tokenizer.decode(tokens)
        else:
            # HuggingFace tokenizer
            decoded = tokenizer.decode(tokens)
        
        decode_time = time.perf_counter() - start_time
        results["decode_times"].append(decode_time)
        
        # Check accuracy (basic check - decoded should contain most of original content)
        # Note: Perfect round-trip may not be expected due to normalization
        if len(decoded.strip()) < len(text.strip()) * 0.8:
            print(f"    Warning: Significant length difference in decoded text")
            print(f"    Original: {len(text)} chars, Decoded: {len(decoded)} chars")
            results["accuracy"] = False
    
    return results

def compare_results(results_list: List[dict]):
    """Compare and display benchmark results."""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS COMPARISON")
    print("="*80)
    
    for results in results_list:
        name = results["name"]
        
        avg_encode_time = statistics.mean(results["encode_times"]) * 1000  # ms
        avg_decode_time = statistics.mean(results["decode_times"]) * 1000  # ms
        total_encode_time = sum(results["encode_times"]) * 1000  # ms
        total_decode_time = sum(results["decode_times"]) * 1000  # ms
        total_tokens = results["total_tokens"]
        
        print(f"\n{name}:")
        print(f"  Total tokens generated: {total_tokens:,}")
        print(f"  Average encode time: {avg_encode_time:.2f} ms")
        print(f"  Average decode time: {avg_decode_time:.2f} ms")
        print(f"  Total encode time: {total_encode_time:.2f} ms")
        print(f"  Total decode time: {total_decode_time:.2f} ms")
        print(f"  Encoding throughput: {total_tokens / (total_encode_time/1000):.0f} tokens/sec")
        print(f"  Decoding throughput: {total_tokens / (total_decode_time/1000):.0f} tokens/sec")
        print(f"  Accuracy check: {'✓ PASS' if results['accuracy'] else '✗ FAIL'}")
    
    # Performance comparison
    if len(results_list) >= 2:
        bpe_qwen_results = next((r for r in results_list if "bpe-qwen" in r["name"]), None)
        hf_results = next((r for r in results_list if "HuggingFace" in r["name"]), None)
        
        if bpe_qwen_results and hf_results:
            print(f"\n{'-'*50}")
            print("PERFORMANCE COMPARISON:")
            
            bpe_encode_total = sum(bpe_qwen_results["encode_times"]) * 1000
            hf_encode_total = sum(hf_results["encode_times"]) * 1000
            encode_speedup = hf_encode_total / bpe_encode_total
            
            bpe_decode_total = sum(bpe_qwen_results["decode_times"]) * 1000
            hf_decode_total = sum(hf_results["decode_times"]) * 1000
            decode_speedup = hf_decode_total / bpe_decode_total
            
            print(f"Encoding speedup: {encode_speedup:.2f}x {'faster' if encode_speedup > 1 else 'slower'}")
            print(f"Decoding speedup: {decode_speedup:.2f}x {'faster' if decode_speedup > 1 else 'slower'}")

def test_accuracy(qwen_tokenizer, hf_tokenizer, texts: List[str]):
    """Test encoding/decoding accuracy between tokenizers."""
    print("\n" + "="*80)
    print("ACCURACY COMPARISON")
    print("="*80)
    
    for i, text in enumerate(texts[:3]):  # Test first 3 texts for accuracy
        print(f"\nTest {i+1}: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        try:
            # Our tokenizer
            qwen_tokens = qwen_tokenizer.encode(text)
            qwen_decoded = qwen_tokenizer.decode(qwen_tokens)
            
            # HuggingFace tokenizer
            hf_tokens = hf_tokenizer(text, return_tensors=None)['input_ids']
            hf_decoded = hf_tokenizer.decode(hf_tokens)
            
            print(f"  bpe-qwen tokens: {len(qwen_tokens)}")
            print(f"  HuggingFace tokens: {len(hf_tokens)}")
            print(f"  Token count difference: {abs(len(qwen_tokens) - len(hf_tokens))}")
            
            # Check if decoded text is similar
            qwen_clean = qwen_decoded.strip()
            hf_clean = hf_decoded.strip()
            
            if qwen_clean == hf_clean:
                print("  ✓ Decoded text matches exactly")
            else:
                print(f"  ⚠ Decoded text differs")
                print(f"    bpe-qwen: {qwen_clean[:100]}{'...' if len(qwen_clean) > 100 else ''}")
                print(f"    HuggingFace: {hf_clean[:100]}{'...' if len(hf_clean) > 100 else ''}")
                
        except Exception as e:
            print(f"  ✗ Error during comparison: {e}")

def main():
    """Main benchmark function."""
    print("BPE-Qwen vs HuggingFace Tokenizer Benchmark")
    print("="*80)
    
    # Install dependencies
    transformers = install_dependencies()
    
    # Create test texts
    texts = create_test_texts()
    print(f"Created {len(texts)} test texts")
    
    try:
        # Initialize our tokenizer
        print("\nInitializing bpe-qwen tokenizer...")
        qwen_tokenizer = bpe_qwen.QwenTokenizer("vocab.json", "merges.txt")
        print(f"✓ bpe-qwen loaded, vocab size: {qwen_tokenizer.vocab_size()}")
        
        # Initialize HuggingFace tokenizer
        print("\nInitializing HuggingFace tokenizer...")
        try:
            hf_tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
            print(f"✓ HuggingFace tokenizer loaded, vocab size: {hf_tokenizer.vocab_size}")
        except Exception as e:
            print(f"✗ Failed to load HuggingFace tokenizer: {e}")
            print("Will try to use local tokenizer files...")
            hf_tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
            print(f"✓ Local HuggingFace tokenizer loaded, vocab size: {hf_tokenizer.vocab_size}")
        
        # Run benchmarks
        results = []
        
        # Benchmark our implementation
        qwen_results = benchmark_tokenizer(qwen_tokenizer, texts, "bpe-qwen (Rust)")
        results.append(qwen_results)
        
        # Benchmark HuggingFace implementation
        hf_results = benchmark_tokenizer(hf_tokenizer, texts, "HuggingFace (Python)")
        results.append(hf_results)
        
        # Compare results
        compare_results(results)
        
        # Test accuracy
        test_accuracy(qwen_tokenizer, hf_tokenizer, texts)
        
    except Exception as e:
        print(f"✗ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()