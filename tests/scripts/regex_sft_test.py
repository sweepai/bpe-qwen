#!/usr/bin/env python3
"""
Script to test automaton vs regex pretokenization on SFT JSONL data at scale.
Based on the Rust test test_automaton_single_pass_matches_regex_on_samples but using real SFT data.
"""

import json
import sys
import os
import re
import time
from pathlib import Path
import typer
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import tokenization functions - fail fast if not available
from bpe_qwen.bpe_qwen import pretokenize_fast_single_pass_indices, pretokenize_fast_single_pass_indices_automaton


def compare_automaton_vs_regex(text, text_name="text"):
    """
    Compare automaton vs regex single-pass pretokenization on the given text.
    This mirrors the Rust test test_automaton_single_pass_matches_regex_on_samples.

    Args:
        text (str): Text to tokenize
        text_name (str): Name/description of the text for output

    Returns:
        dict: Results from both methods, timing info, and whether they match
    """
    # Time regex implementation
    start_time = time.perf_counter()
    expected_indices = pretokenize_fast_single_pass_indices(text)
    regex_time = time.perf_counter() - start_time

    # Time automaton implementation
    start_time = time.perf_counter()
    actual_indices = pretokenize_fast_single_pass_indices_automaton(text)
    automaton_time = time.perf_counter() - start_time

    # Check if they match
    matches = expected_indices == actual_indices

    # Calculate speedup
    speedup = regex_time / automaton_time if automaton_time > 0 else float('inf')

    return {
        'expected_indices': expected_indices,
        'actual_indices': actual_indices,
        'matches': matches,
        'text_name': text_name,
        'text': text,
        'regex_time': regex_time,
        'automaton_time': automaton_time,
        'speedup': speedup
    }


def show_mismatch_details(result):
    """Show detailed information about a mismatch."""
    expected = result['expected_indices']
    actual = result['actual_indices']
    text_name = result['text_name']
    text = result.get('text', '')

    print(f"\n🔍 MISMATCH in {text_name}:")
    print(f"Expected length: {len(expected)}, Actual length: {len(actual)}")

    # Find first difference
    min_len = min(len(expected), len(actual))
    first_diff = -1
    for i in range(min_len):
        if expected[i] != actual[i]:
            first_diff = i
            break

    if first_diff >= 0:
        print(f"First difference at index {first_diff}: expected {expected[first_diff]}, got {actual[first_diff]}")

        # Show context around the mismatch
        print(f"\n📍 Context around mismatch (±5 indices):")
        start_idx = max(0, first_diff - 5)
        end_idx = min(min_len, first_diff + 6)

        print(f"Context range: indices {start_idx} to {end_idx-1}")

        # Show the actual text segments if we have the text
        if text:
            print(f"\n📝 Text analysis around position {expected[first_diff] if first_diff < len(expected) else actual[first_diff]}:")

            # Get the problematic position
            problem_pos = expected[first_diff] if first_diff < len(expected) else actual[first_diff]

            # Show text context around the problem
            context_start = max(0, problem_pos - 50)
            context_end = min(len(text), problem_pos + 50)
            context_text = text[context_start:context_end]

            print(f"Text context: {repr(context_text)}")
            print(f"Problem position: {problem_pos} (relative to context: {problem_pos - context_start})")

            # Show what each implementation thinks should be the token boundaries
            if first_diff > 0:
                prev_expected = expected[first_diff - 1] if first_diff - 1 < len(expected) else 0
                prev_actual = actual[first_diff - 1] if first_diff - 1 < len(actual) else 0
            else:
                prev_expected = prev_actual = 0

            curr_expected = expected[first_diff] if first_diff < len(expected) else len(text)
            curr_actual = actual[first_diff] if first_diff < len(actual) else len(text)

            print(f"\nExpected token: {repr(text[prev_expected:curr_expected])}")
            print(f"Actual token:   {repr(text[prev_actual:curr_actual])}")

    elif len(expected) != len(actual):
        print(f"Length difference: expected has {len(expected)} items, actual has {len(actual)} items")


def analyze_automaton_vs_regex(result, verbose=False):
    """
    Analyze differences between automaton and regex methods.

    Args:
        result (dict): Results from compare_automaton_vs_regex
        verbose (bool): Whether to show detailed output even on success

    Returns:
        bool: True if there was a mismatch, False if they matched
    """
    if not result['matches']:
        show_mismatch_details(result)
        return True
    else:
        if verbose:
            print(f"✅ {result['text_name']}: automaton matches regex implementation ({len(result['expected_indices'])} indices)")
            print(f"   ⏱️  Timing: regex={result['regex_time']:.6f}s, automaton={result['automaton_time']:.6f}s, speedup={result['speedup']:.2f}x")
        return False


def load_and_test_jsonl(file_path, max_entries=0, verbose=False):
    """
    Load JSONL file and test automaton vs regex implementation on each entry.

    Args:
        file_path (str): Path to the JSONL file
        max_entries (int): Maximum number of entries to process (0 for all)
        verbose (bool): Whether to show detailed output for all entries
    """
    if verbose:
        print(f"Loading JSONL file: {file_path}")
        print("Testing automaton vs regex implementation at scale...")

    # First pass: count total lines for progress bar
    total_lines = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                total_lines += 1

    # Determine how many entries we'll actually process
    entries_to_process = min(total_lines, max_entries) if max_entries > 0 else total_lines

    total_entries = 0
    processed_entries = 0
    mismatches = 0
    total_texts_tested = 0

    # Speed tracking variables
    total_regex_time = 0.0
    total_automaton_time = 0.0
    speedup_measurements = []

    # Fail fast - let file operations fail immediately if there are issues
    with open(file_path, 'r', encoding='utf-8') as f:
        # Use tqdm for progress tracking
        progress_bar = tqdm(total=entries_to_process, desc="Testing entries", disable=verbose)

        try:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                total_entries += 1

                # Fail fast - let JSON parsing fail immediately
                data = json.loads(line)

                # Fail fast - require both fields to be present
                prompt = data['prompt']
                completion = data['completion']

                entry_has_mismatch = False

                # Test prompt
                prompt_result = compare_automaton_vs_regex(prompt, f"entry {processed_entries + 1} prompt")
                prompt_mismatch = analyze_automaton_vs_regex(prompt_result, verbose=verbose)
                total_texts_tested += 1

                # Accumulate timing data
                total_regex_time += prompt_result['regex_time']
                total_automaton_time += prompt_result['automaton_time']
                speedup_measurements.append(prompt_result['speedup'])

                # Test completion
                completion_result = compare_automaton_vs_regex(completion, f"entry {processed_entries + 1} completion")
                completion_mismatch = analyze_automaton_vs_regex(completion_result, verbose=verbose)
                total_texts_tested += 1

                # Accumulate timing data
                total_regex_time += completion_result['regex_time']
                total_automaton_time += completion_result['automaton_time']
                speedup_measurements.append(completion_result['speedup'])

                entry_has_mismatch = prompt_mismatch or completion_mismatch
                if entry_has_mismatch:
                    mismatches += 1

                # Only show detailed entry info if there's a mismatch or verbose mode
                if entry_has_mismatch or verbose:
                    print(f"\n--- Entry {processed_entries + 1} (Line {line_num}) ---")
                    print(f"Prompt length: {len(prompt)} characters")
                    print(f"Completion length: {len(completion)} characters")
                    if entry_has_mismatch:
                        print("⚠️  AUTOMATON/REGEX IMPLEMENTATION MISMATCH DETECTED")

                processed_entries += 1
                progress_bar.update(1)

                # Optional: limit output for large files
                if max_entries > 0 and processed_entries >= max_entries:
                    if verbose:
                        print(f"\n... (showing first {max_entries} entries, total: {total_entries})")
                    break

        finally:
            progress_bar.close()

    # Always show final summary
    print(f"\n{'='*60}")
    print(f"AUTOMATON VS REGEX IMPLEMENTATION TEST RESULTS")
    print(f"{'='*60}")
    print(f"Total entries found: {total_entries}")
    print(f"Successfully processed: {processed_entries}")
    print(f"Total texts tested: {total_texts_tested}")

    if mismatches == 0:
        print(f"✅ ALL {total_texts_tested} TEXTS PASSED: Automaton implementation matches regex implementation perfectly!")
    else:
        print(f"⚠️  MISMATCHES FOUND: {mismatches} entries had automaton/regex implementation differences")
        print(f"Success rate: {((total_texts_tested - mismatches) / total_texts_tested * 100):.2f}%")

    # Speed comparison summary
    if speedup_measurements:
        print(f"\n{'='*60}")
        print(f"SPEED COMPARISON RESULTS")
        print(f"{'='*60}")
        print(f"Total regex time: {total_regex_time:.6f}s")
        print(f"Total automaton time: {total_automaton_time:.6f}s")
        overall_speedup = total_regex_time / total_automaton_time if total_automaton_time > 0 else float('inf')
        print(f"Overall speedup: {overall_speedup:.2f}x")

        # Calculate statistics
        avg_speedup = sum(speedup_measurements) / len(speedup_measurements)
        min_speedup = min(speedup_measurements)
        max_speedup = max(speedup_measurements)

        print(f"Average speedup per text: {avg_speedup:.2f}x")
        print(f"Min speedup: {min_speedup:.2f}x")
        print(f"Max speedup: {max_speedup:.2f}x")

        # Performance interpretation
        if overall_speedup > 1.0:
            print(f"🚀 Automaton is {overall_speedup:.2f}x FASTER than regex implementation")
        elif overall_speedup < 1.0:
            print(f"🐌 Automaton is {1/overall_speedup:.2f}x SLOWER than regex implementation")
        else:
            print(f"⚖️  Automaton and regex have similar performance")


def main(
    file_path: str = typer.Argument(..., help="Path to the JSONL file"),
    max_entries: int = typer.Option(0, "--max-entries", help="Maximum number of entries to process (0 for all)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output for all entries"),
):
    """Test automaton vs regex implementation pretokenization on SFT JSONL data at scale."""

    if verbose:
        print("Automaton vs Regex Implementation Pretokenization Test")
        print("=" * 60)
        print("This test compares pretokenize_fast_single_pass_indices_automaton")
        print("with pretokenize_fast_single_pass_indices on real SFT data.")
        print("Based on: test_automaton_single_pass_matches_regex_on_samples")

    # Fail fast - let any errors bubble up immediately
    load_and_test_jsonl(
        file_path=file_path,
        max_entries=max_entries,
        verbose=verbose
    )


if __name__ == "__main__":
    typer.run(main)