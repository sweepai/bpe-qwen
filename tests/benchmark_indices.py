#!/usr/bin/env python3
"""
Benchmark comparing string-based vs indices-based pretokenization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bpe_qwen.bpe_qwen import pretokenize_fast, pretokenize_fast_indices, indices_to_strings, pretokenize_slow
import time

def benchmark_pretokenization():
    """Compare performance of different pretokenization approaches."""

    # Test texts of various sizes
    test_texts = {
        "small": "Hello, world! This is a test.",

        "medium": """Natural language processing (NLP) is a subfield of linguistics, computer science, 
        and artificial intelligence concerned with the interactions between computers and human language, 
        in particular how to program computers to process and analyze large amounts of natural language data. 
        The goal is a computer capable of understanding the contents of documents, including the contextual 
        nuances of the language within them.""" * 2,

        "large": """def process_data(input_file, output_file, config=None):
            '''Process data from input file and write to output file.'''
            import json
            import pandas as pd
            from pathlib import Path

            # Default configuration
            default_config = {
                'chunk_size': 1000,
                'encoding': 'utf-8',
                'compression': 'gzip',
                'verbose': True
            }""" * 50,
    }

    print("=" * 80)
    print("PRETOKENIZATION BENCHMARK: String-based vs Indices-based")
    print("=" * 80)

    for name, text in test_texts.items():
        print(f"\n{name.upper()} text ({len(text)} chars)")
        print("-" * 40)

        # Warm up
        _ = pretokenize_slow(text)
        _ = pretokenize_fast(text)
        indices = pretokenize_fast_indices(text)
        _ = indices_to_strings(text, indices)

        # Benchmark slow (fancy-regex)
        iterations = 100 if len(text) < 1000 else 50

        start = time.perf_counter()
        for _ in range(iterations):
            slow_result = pretokenize_slow(text)
        slow_time = (time.perf_counter() - start) / iterations * 1000

        # Benchmark fast string-based
        start = time.perf_counter()
        for _ in range(iterations):
            fast_result = pretokenize_fast(text)
        fast_time = (time.perf_counter() - start) / iterations * 1000

        # Benchmark indices-based (without conversion)
        start = time.perf_counter()
        for _ in range(iterations):
            indices_result = pretokenize_fast_indices(text)
        indices_time = (time.perf_counter() - start) / iterations * 1000

        # Benchmark indices-based (with conversion to strings)
        start = time.perf_counter()
        for _ in range(iterations):
            indices = pretokenize_fast_indices(text)
            strings = indices_to_strings(text, indices)
        indices_with_conversion_time = (time.perf_counter() - start) / iterations * 1000

        # Convert indices result for comparison
        indices_strings = indices_to_strings(text, indices_result)

        # Print results
        print(f"  Slow (fancy-regex):       {slow_time:.3f} ms")
        print(f"  Fast (string-based):      {fast_time:.3f} ms  (speedup: {slow_time/fast_time:.2f}x)")
        print(f"  Indices (no conversion):  {indices_time:.3f} ms  (speedup: {slow_time/indices_time:.2f}x)")
        print(f"  Indices (with strings):   {indices_with_conversion_time:.3f} ms  (speedup: {slow_time/indices_with_conversion_time:.2f}x)")

        # Check correctness
        if slow_result == fast_result == indices_strings:
            print(f"  Correctness: ✓ All methods match ({len(slow_result)} tokens)")
        else:
            print(f"  Correctness: ✗ MISMATCH!")
            print(f"    Slow: {len(slow_result)} tokens")
            print(f"    Fast: {len(fast_result)} tokens")
            print(f"    Indices: {len(indices_strings)} tokens")

            # Show first differences
            for i in range(min(5, max(len(slow_result), len(fast_result), len(indices_strings)))):
                s = slow_result[i] if i < len(slow_result) else "N/A"
                f = fast_result[i] if i < len(fast_result) else "N/A"
                idx = indices_strings[i] if i < len(indices_strings) else "N/A"
                if s != f or s != idx:
                    print(f"    Position {i}: slow={repr(s)}, fast={repr(f)}, indices={repr(idx)}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("The indices-based approach avoids string allocations in the core algorithm.")
    print("Even with conversion back to strings, it can be faster due to:")
    print("1. More efficient boundary adjustments (just changing indices)")
    print("2. Better cache locality (working with numbers instead of strings)")
    print("3. Reduced allocator pressure")

if __name__ == "__main__":
    benchmark_pretokenization()