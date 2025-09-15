#!/usr/bin/env python3
"""
Script to load and process SFT JSONL data with prompt/completion format.
Processes every entry in the JSONL file and compares tokenization methods.
"""

import json
import sys
import os
from pathlib import Path
import typer
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import tokenization functions - fail fast if not available
from bpe_qwen.bpe_qwen import pretokenize_slow, pretokenize_fast
from bpe_qwen.bpe_qwen import pretokenize_fast_indices, indices_to_strings


def compare_tokenization_methods(text, text_name="text"):
    """
    Compare different tokenization methods on the given text.

    Args:
        text (str): Text to tokenize
        text_name (str): Name/description of the text for output

    Returns:
        dict: Results from different tokenization methods
    """
    # Fail fast - let any errors bubble up immediately
    slow_result = pretokenize_slow(text)
    fast_result = pretokenize_fast(text)

    indices = pretokenize_fast_indices(text)
    indices_result = indices_to_strings(text, indices)

    return {
        'slow': slow_result,
        'fast': fast_result,
        'indices': indices_result
    }


def find_first_mismatch_position(list1, list2):
    """Find the position of the first mismatch between two lists."""
    min_len = min(len(list1), len(list2))
    for i in range(min_len):
        if list1[i] != list2[i]:
            return i
    # If all compared elements match but lengths differ, mismatch is at min_len
    if len(list1) != len(list2):
        return min_len
    return -1  # No mismatch found


def show_mismatch_context(list1, list2, name1, name2, context_size=3):
    """Show context around the first mismatch between two token lists."""
    mismatch_pos = find_first_mismatch_position(list1, list2)
    if mismatch_pos == -1:
        return

    print(f"\n🔍 First mismatch at position {mismatch_pos} ({name1} vs {name2}):")

    # Calculate context window
    start = max(0, mismatch_pos - context_size)
    end1 = min(len(list1), mismatch_pos + context_size + 1)
    end2 = min(len(list2), mismatch_pos + context_size + 1)

    # Show context for first list
    context1 = list1[start:end1]
    print(f"{name1:>8}: {context1}")

    # Show context for second list
    context2 = list2[start:end2]
    print(f"{name2:>8}: {context2}")

    # Show pointer to mismatch position
    pointer_pos = mismatch_pos - start
    if pointer_pos < len(context1) and pointer_pos < len(context2):
        spaces = " " * (11 + sum(len(repr(token)) + 2 for token in context1[:pointer_pos]))
        print(f"{spaces}^ mismatch here")


def analyze_tokenization_differences(results, text_name="text", verbose=False):
    """
    Analyze differences between tokenization methods.

    Args:
        results (dict): Results from compare_tokenization_methods
        text_name (str): Name/description of the text for output
        verbose (bool): Whether to show detailed output even on success
    """
    # Fail fast - expect all results to be present
    slow = results['slow']
    fast = results['fast']
    indices = results['indices']

    # Check for mismatches and show context
    mismatches = []
    has_mismatch = False

    if slow != fast:
        mismatches.append("slow vs fast")
        if not has_mismatch:
            show_mismatch_context(slow, fast, "slow", "fast")
            has_mismatch = True

    if fast != indices:
        mismatches.append("fast vs indices")
        if not has_mismatch:
            show_mismatch_context(fast, indices, "fast", "indices")
            has_mismatch = True

    if slow != indices:
        mismatches.append("slow vs indices")
        if not has_mismatch:
            show_mismatch_context(slow, indices, "slow", "indices")
            has_mismatch = True

    if mismatches:
        print(f"\n--- Tokenization Analysis for {text_name} ---")
        print(f"Slow tokens ({len(slow)}): {slow[:5]}{'...' if len(slow) > 5 else ''}")
        print(f"Fast tokens ({len(fast)}): {fast[:5]}{'...' if len(fast) > 5 else ''}")
        print(f"Indices tokens ({len(indices)}): {indices[:5]}{'...' if len(indices) > 5 else ''}")
        print(f"⚠️  TOKENIZATION MISMATCHES: {', '.join(mismatches)}")
        return True
    else:
        if verbose:
            print(f"\n--- Tokenization Analysis for {text_name} ---")
            print(f"Slow tokens ({len(slow)}): {slow[:5]}{'...' if len(slow) > 5 else ''}")
            print(f"Fast tokens ({len(fast)}): {fast[:5]}{'...' if len(fast) > 5 else ''}")
            print(f"Indices tokens ({len(indices)}): {indices[:5]}{'...' if len(indices) > 5 else ''}")
            print("✅ All tokenization methods match")
        return False


def load_and_process_jsonl(file_path, compare_tokenization=True, max_entries=0, verbose=False, break_on_error=True):
    """
    Load JSONL file and process each entry with prompt/completion format.

    Args:
        file_path (str): Path to the JSONL file
        compare_tokenization (bool): Whether to compare tokenization methods
        max_entries (int): Maximum number of entries to process (0 for all)
        verbose (bool): Whether to show detailed output for all entries
        break_on_error (bool): Whether to stop on first tokenization mismatch
    """
    if verbose:
        print(f"Loading JSONL file: {file_path}")

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
    tokenization_mismatches = 0

    # Fail fast - let file operations fail immediately if there are issues
    with open(file_path, 'r', encoding='utf-8') as f:
        # Use tqdm for progress tracking
        progress_bar = tqdm(total=entries_to_process, desc="Processing entries", disable=verbose)

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

                # Compare tokenization methods if requested
                if compare_tokenization:
                    # Tokenize prompt - fail fast on any tokenization errors
                    prompt_results = compare_tokenization_methods(prompt, f"prompt {processed_entries + 1}")
                    prompt_mismatch = analyze_tokenization_differences(prompt_results, f"prompt {processed_entries + 1}", verbose=verbose)

                    # Tokenize completion - fail fast on any tokenization errors
                    completion_results = compare_tokenization_methods(completion, f"completion {processed_entries + 1}")
                    completion_mismatch = analyze_tokenization_differences(completion_results, f"completion {processed_entries + 1}", verbose=verbose)

                    entry_has_mismatch = prompt_mismatch or completion_mismatch
                    if entry_has_mismatch:
                        tokenization_mismatches += 1

                # Only show detailed entry info if there's a mismatch or verbose mode
                if entry_has_mismatch or verbose:
                    print(f"\n--- Entry {processed_entries + 1} (Line {line_num}) ---")
                    print(f"Prompt length: {len(prompt)} characters")
                    print(f"Completion length: {len(completion)} characters")
                    print(f"Prompt preview: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
                    print(f"Completion preview: {completion[:100]}{'...' if len(completion) > 100 else ''}")

                processed_entries += 1
                progress_bar.update(1)

                # Break on first error if requested
                if break_on_error and entry_has_mismatch:
                    progress_bar.close()
                    print(f"\n❌ Stopping on first tokenization mismatch (entry {processed_entries})")
                    return

                # Optional: limit output for large files
                if max_entries > 0 and processed_entries >= max_entries:
                    if verbose:
                        print(f"\n... (showing first {max_entries} entries, total: {total_entries})")
                    break

        finally:
            progress_bar.close()

    # Always show final summary
    if compare_tokenization:
        if tokenization_mismatches == 0:
            print(f"✅ All {processed_entries} entries passed tokenization validation")
        else:
            print(f"\nProcessing complete!")
            print(f"Total entries found: {total_entries}")
            print(f"Successfully processed: {processed_entries}")
            print(f"Tokenization mismatches found: {tokenization_mismatches}/{processed_entries} entries")
    else:
        print(f"Processed {processed_entries} entries successfully")


def main(
    file_path: str = typer.Argument(..., help="Path to the JSONL file"),
    no_tokenization: bool = typer.Option(False, "--no-tokenization", help="Skip tokenization comparison"),
    max_entries: int = typer.Option(0, "--max-entries", help="Maximum number of entries to process (0 for all)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output for all entries"),
    break_on_error: bool = typer.Option(True, "--break-on-error/--no-break-on-error", help="Stop processing on first tokenization mismatch")
):
    """Process SFT JSONL data and compare tokenization methods."""

    if verbose:
        print("SFT JSONL Data Processor with Tokenization Comparison")
        print("=" * 60)

        if no_tokenization:
            print("Tokenization comparison disabled")

    # Fail fast - let any errors bubble up immediately
    load_and_process_jsonl(
        file_path=file_path,
        compare_tokenization=not no_tokenization,
        max_entries=max_entries,
        verbose=verbose,
        break_on_error=break_on_error
    )


if __name__ == "__main__":
    typer.run(main)