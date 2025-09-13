#!/usr/bin/env python3
"""
Benchmark comparing all pretokenization approaches including the automata.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bpe_qwen.bpe_qwen import (
    pretokenize_slow,
    pretokenize_fast,
    pretokenize_automata,
    pretokenize_fast_indices,
    indices_to_strings
)
import time

def benchmark_all_approaches():
    """Compare performance and correctness of all pretokenization approaches."""

    # Test texts of various sizes and complexities
    test_texts = {
        "small": "Hello, world! This is a test.",

        "contractions": "I've got what you're looking for. We'll see if it's working.",

        "numbers": "    1234 test 5678",

        "punctuation": "Test! Really? Yes... (maybe) [brackets] {braces} <tags>",

        "whitespace_heavy": "   lots    of     spaces     here   ",

        "crlf": "line1\rline2\nline3\r\nline4",

        "medium": """Natural language processing (NLP) is a subfield of linguistics, computer science, 
        and artificial intelligence concerned with the interactions between computers and human language, 
        in particular how to program computers to process and analyze large amounts of natural language data. 
        The goal is a computer capable of understanding the contents of documents, including the contextual 
        nuances of the language within them.""",

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
            }

            if config:
                default_config.update(config)

            # Read input file
            input_path = Path(input_file)
            if not input_path.exists():
                raise FileNotFoundError(f"Input file {input_file} not found")

            # Process based on file extension
            if input_path.suffix == '.json':
                with open(input_path, 'r', encoding=default_config['encoding']) as f:
                    data = json.load(f)
            elif input_path.suffix == '.csv':
                data = pd.read_csv(input_path, encoding=default_config['encoding'])

            return data""" * 10,
    }

    print("=" * 80)
    print("PRETOKENIZATION BENCHMARK: All Approaches")
    print("=" * 80)

    # Summary stats
    total_matches = 0
    total_mismatches = 0
    approach_times = {
        'slow': [],
        'fast': [],
        'automata': [],
        'indices': []
    }

    for name, text in test_texts.items():
        print(f"\n{name.upper()} text ({len(text)} chars)")
        print("-" * 40)

        # Warm up
        _ = pretokenize_slow(text)
        _ = pretokenize_fast(text)
        _ = pretokenize_automata(text)
        indices = pretokenize_fast_indices(text)
        _ = indices_to_strings(text, indices)

        # Determine iterations based on text size
        iterations = 100 if len(text) < 500 else 50 if len(text) < 5000 else 20

        # Benchmark slow (fancy-regex) - baseline
        start = time.perf_counter()
        for _ in range(iterations):
            slow_result = pretokenize_slow(text)
        slow_time = (time.perf_counter() - start) / iterations * 1000
        approach_times['slow'].append(slow_time)

        # Benchmark fast (two-pass)
        start = time.perf_counter()
        for _ in range(iterations):
            fast_result = pretokenize_fast(text)
        fast_time = (time.perf_counter() - start) / iterations * 1000
        approach_times['fast'].append(fast_time)

        # Benchmark automata (single-pass, no regex)
        start = time.perf_counter()
        for _ in range(iterations):
            automata_result = pretokenize_automata(text)
        automata_time = (time.perf_counter() - start) / iterations * 1000
        approach_times['automata'].append(automata_time)

        # Benchmark indices (with conversion)
        start = time.perf_counter()
        for _ in range(iterations):
            indices = pretokenize_fast_indices(text)
            indices_result = indices_to_strings(text, indices)
        indices_time = (time.perf_counter() - start) / iterations * 1000
        approach_times['indices'].append(indices_time)

        # Print results
        print(f"  Slow (fancy-regex):    {slow_time:.3f} ms  (baseline)")
        print(f"  Fast (two-pass):       {fast_time:.3f} ms  (speedup: {slow_time/fast_time:.2f}x)")
        print(f"  Automata (single):     {automata_time:.3f} ms  (speedup: {slow_time/automata_time:.2f}x)")
        print(f"  Indices (w/ convert):  {indices_time:.3f} ms  (speedup: {slow_time/indices_time:.2f}x)")

        # Check correctness
        all_match = (slow_result == fast_result == automata_result == indices_result)

        if all_match:
            print(f"  Correctness: ✓ All methods match ({len(slow_result)} tokens)")
            total_matches += 1
        else:
            print(f"  Correctness: ✗ MISMATCH!")
            total_mismatches += 1

            # Show differences
            print(f"    Slow:     {len(slow_result)} tokens")
            print(f"    Fast:     {len(fast_result)} tokens")
            print(f"    Automata: {len(automata_result)} tokens")
            print(f"    Indices:  {len(indices_result)} tokens")

            # Show first few differences
            max_len = max(len(slow_result), len(fast_result), len(automata_result), len(indices_result))
            diffs_shown = 0
            for i in range(min(max_len, 100)):
                s = slow_result[i] if i < len(slow_result) else "N/A"
                f = fast_result[i] if i < len(fast_result) else "N/A"
                a = automata_result[i] if i < len(automata_result) else "N/A"
                idx = indices_result[i] if i < len(indices_result) else "N/A"

                if not (s == f == a == idx):
                    print(f"    Position {i}: slow={repr(s)}, fast={repr(f)}, automata={repr(a)}, indices={repr(idx)}")
                    diffs_shown += 1
                    if diffs_shown >= 5:
                        print(f"    ... and {max_len - i - 1} more tokens")
                        break

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nCorrectness: {total_matches}/{total_matches + total_mismatches} tests passed")

    print("\nAverage Performance (ms):")
    for approach, times in approach_times.items():
        avg_time = sum(times) / len(times)
        print(f"  {approach:12s}: {avg_time:.3f} ms")

    # Calculate average speedups
    slow_avg = sum(approach_times['slow']) / len(approach_times['slow'])
    fast_avg = sum(approach_times['fast']) / len(approach_times['fast'])
    automata_avg = sum(approach_times['automata']) / len(approach_times['automata'])
    indices_avg = sum(approach_times['indices']) / len(approach_times['indices'])

    print("\nAverage Speedup vs Slow (fancy-regex):")
    print(f"  Fast (two-pass):      {slow_avg/fast_avg:.2f}x")
    print(f"  Automata (single):    {slow_avg/automata_avg:.2f}x")
    print(f"  Indices (w/ convert): {slow_avg/indices_avg:.2f}x")

    print("\nKey Observations:")
    print("- The automata approach implements the pattern in a single pass")
    print("- No regex compilation overhead or second correction pass needed")
    print("- Direct state machine implementation can be very efficient")
    print("- Trade-off: More complex code but potentially better performance")

if __name__ == "__main__":
    benchmark_all_approaches()