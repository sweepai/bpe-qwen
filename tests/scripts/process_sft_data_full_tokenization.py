#!/usr/bin/env python3
"""
Script to load and process SFT JSONL data with prompt/completion format.
Processes every entry in the JSONL file using full tokenization with AutoLinearTokenizer.
"""

import json
import sys
import os
from pathlib import Path
import typer
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import AutoLinearTokenizer - fail fast if not available
from bpe_qwen.auto_linear_tokenizer import AutoLinearTokenizer

def install_transformers():
    """Install transformers if needed."""
    try:
        import transformers
        return transformers
    except ImportError:
        print("Installing transformers...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
        import transformers
        return transformers


def compare_tokenization_methods(auto_linear_tokenizer, hf_tokenizer, text, text_name="text"):
    """
    Compare AutoLinearTokenizer against HuggingFace tokenizer.

    Args:
        auto_linear_tokenizer: AutoLinearTokenizer instance
        hf_tokenizer: HuggingFace tokenizer instance
        text (str): Text to tokenize
        text_name (str): Name/description of the text for output

    Returns:
        dict: Results from both tokenizers and comparison
    """
    # Fail fast - let any errors bubble up immediately

    # AutoLinearTokenizer results
    auto_token_ids = auto_linear_tokenizer.encode(text)
    auto_decoded = auto_linear_tokenizer.decode(auto_token_ids)
    auto_tokens = [auto_linear_tokenizer.decode([token_id]) for token_id in auto_token_ids]

    # HuggingFace tokenizer results
    hf_token_ids = hf_tokenizer(text, return_tensors=None, add_special_tokens=False)['input_ids']
    hf_decoded = hf_tokenizer.decode(hf_token_ids, skip_special_tokens=False)
    hf_tokens = [hf_tokenizer.decode([token_id]) for token_id in hf_token_ids]

    # Check for mismatches
    tokens_match = auto_token_ids == hf_token_ids
    decoded_match = auto_decoded == hf_decoded or auto_decoded.strip() == hf_decoded.strip()

    return {
        'auto_linear': {
            'token_ids': auto_token_ids,
            'tokens': auto_tokens,
            'decoded_text': auto_decoded,
            'token_count': len(auto_token_ids),
        },
        'huggingface': {
            'token_ids': hf_token_ids,
            'tokens': hf_tokens,
            'decoded_text': hf_decoded,
            'token_count': len(hf_token_ids),
        },
        'comparison': {
            'tokens_match': tokens_match,
            'decoded_match': decoded_match,
            'has_mismatch': not (tokens_match and decoded_match),
        },
        'char_count': len(text)
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
    print(f"{name1:>12}: {context1}")

    # Show context for second list
    context2 = list2[start:end2]
    print(f"{name2:>12}: {context2}")

    # Show pointer to mismatch position
    pointer_pos = mismatch_pos - start
    if pointer_pos < len(context1) and pointer_pos < len(context2):
        spaces = " " * (15 + sum(len(repr(token)) + 2 for token in context1[:pointer_pos]))
        print(f"{spaces}^ mismatch here")


def analyze_tokenization_results(results, text_name="text", verbose=False):
    """
    Analyze tokenization comparison results and show statistics.

    Args:
        results (dict): Results from compare_tokenization_methods
        text_name (str): Name/description of the text for output
        verbose (bool): Whether to show detailed output
    """
    auto_results = results['auto_linear']
    hf_results = results['huggingface']
    comparison = results['comparison']
    char_count = results['char_count']

    # Calculate compression ratios
    auto_compression = char_count / auto_results['token_count'] if auto_results['token_count'] > 0 else 0
    hf_compression = char_count / hf_results['token_count'] if hf_results['token_count'] > 0 else 0

    has_mismatch = comparison['has_mismatch']

    if has_mismatch or verbose:
        print(f"\n--- Tokenization Analysis for {text_name} ---")
        print(f"Character count: {char_count}")
        print(f"AutoLinear tokens ({auto_results['token_count']}): {auto_results['tokens'][:5]}{'...' if len(auto_results['tokens']) > 5 else ''}")
        print(f"HuggingFace tokens ({hf_results['token_count']}): {hf_results['tokens'][:5]}{'...' if len(hf_results['tokens']) > 5 else ''}")
        print(f"AutoLinear compression: {auto_compression:.2f} chars/token")
        print(f"HuggingFace compression: {hf_compression:.2f} chars/token")

        if has_mismatch:
            print(f"⚠️  TOKENIZATION MISMATCH DETECTED!")
            if not comparison['tokens_match']:
                print("   - Token IDs differ")
                show_mismatch_context(auto_results['token_ids'], hf_results['token_ids'],
                                    "AutoLinear", "HuggingFace")
            if not comparison['decoded_match']:
                print("   - Decoded text differs")
                print(f"   AutoLinear decoded: {repr(auto_results['decoded_text'][:100])}")
                print(f"   HuggingFace decoded: {repr(hf_results['decoded_text'][:100])}")
        else:
            print("✅ Tokenization methods match")

    return {
        'auto_compression': auto_compression,
        'hf_compression': hf_compression,
        'auto_token_count': auto_results['token_count'],
        'hf_token_count': hf_results['token_count'],
        'char_count': char_count,
        'has_mismatch': has_mismatch
    }


def load_and_process_jsonl(file_path, tokenize_text=True, max_entries=0, verbose=False):
    """
    Load JSONL file and process each entry with prompt/completion format using AutoLinearTokenizer.

    Args:
        file_path (str): Path to the JSONL file
        tokenize_text (bool): Whether to perform tokenization
        max_entries (int): Maximum number of entries to process (0 for all)
        verbose (bool): Whether to show detailed output for all entries
    """
    if verbose:
        print(f"Loading JSONL file: {file_path}")

    # Initialize tokenizers
    auto_tokenizer = None
    hf_tokenizer = None
    if tokenize_text:
        print("Initializing tokenizers...")

        # Initialize AutoLinearTokenizer
        try:
            auto_tokenizer = AutoLinearTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
            print(f"✓ AutoLinearTokenizer loaded, vocab size: {auto_tokenizer.vocab_size}")
        except Exception as e:
            print(f"✗ Failed to load AutoLinearTokenizer: {e}")
            return

        # Initialize HuggingFace tokenizer for comparison
        transformers = install_transformers()
        try:
            # Try local file first
            hf_tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
            print(f"✓ HuggingFace tokenizer loaded from local file, vocab size: {hf_tokenizer.vocab_size}")
        except:
            try:
                # Fall back to downloading
                hf_tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
                print(f"✓ HuggingFace tokenizer loaded from hub, vocab size: {hf_tokenizer.vocab_size}")
            except Exception as e:
                print(f"✗ Failed to load HuggingFace tokenizer: {e}")
                return

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

    # Statistics tracking
    total_auto_prompt_tokens = 0
    total_auto_completion_tokens = 0
    total_hf_prompt_tokens = 0
    total_hf_completion_tokens = 0
    total_prompt_chars = 0
    total_completion_chars = 0
    auto_compression_ratios = []
    hf_compression_ratios = []

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
                if tokenize_text and auto_tokenizer and hf_tokenizer:
                    # Compare tokenization for prompt - fail fast on any tokenization errors
                    prompt_results = compare_tokenization_methods(auto_tokenizer, hf_tokenizer, prompt, f"prompt {processed_entries + 1}")
                    prompt_analysis = analyze_tokenization_results(prompt_results, f"prompt {processed_entries + 1}", verbose=verbose)

                    # Compare tokenization for completion - fail fast on any tokenization errors
                    completion_results = compare_tokenization_methods(auto_tokenizer, hf_tokenizer, completion, f"completion {processed_entries + 1}")
                    completion_analysis = analyze_tokenization_results(completion_results, f"completion {processed_entries + 1}", verbose=verbose)

                    entry_has_mismatch = prompt_analysis['has_mismatch'] or completion_analysis['has_mismatch']
                    if entry_has_mismatch:
                        tokenization_mismatches += 1

                    # Update statistics
                    total_auto_prompt_tokens += prompt_analysis['auto_token_count']
                    total_auto_completion_tokens += completion_analysis['auto_token_count']
                    total_hf_prompt_tokens += prompt_analysis['hf_token_count']
                    total_hf_completion_tokens += completion_analysis['hf_token_count']
                    total_prompt_chars += prompt_analysis['char_count']
                    total_completion_chars += completion_analysis['char_count']
                    auto_compression_ratios.extend([prompt_analysis['auto_compression'], completion_analysis['auto_compression']])
                    hf_compression_ratios.extend([prompt_analysis['hf_compression'], completion_analysis['hf_compression']])

                # Only show detailed entry info if there's a mismatch or verbose mode
                if entry_has_mismatch or verbose:
                    print(f"\n--- Entry {processed_entries + 1} (Line {line_num}) ---")
                    print(f"Prompt length: {len(prompt)} characters")
                    print(f"Completion length: {len(completion)} characters")
                    print(f"Prompt preview: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
                    print(f"Completion preview: {completion[:100]}{'...' if len(completion) > 100 else ''}")
                    if entry_has_mismatch:
                        print("⚠️  This entry contains tokenization mismatches!")

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
    if tokenize_text and auto_tokenizer and hf_tokenizer:
        if tokenization_mismatches == 0:
            print(f"✅ All {processed_entries} entries passed tokenization validation")
        else:
            print(f"\nProcessing complete!")
            print(f"Total entries found: {total_entries}")
            print(f"Successfully processed: {processed_entries}")
            print(f"Tokenization mismatches found: {tokenization_mismatches}/{processed_entries} entries")

        print(f"\n--- Tokenization Comparison Statistics ---")
        print(f"AutoLinear prompt tokens: {total_auto_prompt_tokens:,}")
        print(f"AutoLinear completion tokens: {total_auto_completion_tokens:,}")
        print(f"AutoLinear total tokens: {total_auto_prompt_tokens + total_auto_completion_tokens:,}")
        print(f"HuggingFace prompt tokens: {total_hf_prompt_tokens:,}")
        print(f"HuggingFace completion tokens: {total_hf_completion_tokens:,}")
        print(f"HuggingFace total tokens: {total_hf_prompt_tokens + total_hf_completion_tokens:,}")
        print(f"Total prompt characters: {total_prompt_chars:,}")
        print(f"Total completion characters: {total_completion_chars:,}")
        print(f"Total characters: {total_prompt_chars + total_completion_chars:,}")

        if auto_compression_ratios and hf_compression_ratios:
            auto_avg_compression = sum(auto_compression_ratios) / len(auto_compression_ratios)
            hf_avg_compression = sum(hf_compression_ratios) / len(hf_compression_ratios)
            auto_overall_compression = (total_prompt_chars + total_completion_chars) / (total_auto_prompt_tokens + total_auto_completion_tokens)
            hf_overall_compression = (total_prompt_chars + total_completion_chars) / (total_hf_prompt_tokens + total_hf_completion_tokens)

            print(f"\nAutoLinear average compression: {auto_avg_compression:.2f} chars/token")
            print(f"HuggingFace average compression: {hf_avg_compression:.2f} chars/token")
            print(f"AutoLinear overall compression: {auto_overall_compression:.2f} chars/token")
            print(f"HuggingFace overall compression: {hf_overall_compression:.2f} chars/token")

            token_diff = (total_auto_prompt_tokens + total_auto_completion_tokens) - (total_hf_prompt_tokens + total_hf_completion_tokens)
            if token_diff != 0:
                print(f"Token count difference: {token_diff:+,} tokens (AutoLinear vs HuggingFace)")
    else:
        print(f"Processed {processed_entries} entries successfully")


def main(
    file_path: str = typer.Argument(..., help="Path to the JSONL file"),
    no_tokenization: bool = typer.Option(False, "--no-tokenization", help="Skip tokenization"),
    max_entries: int = typer.Option(0, "--max-entries", help="Maximum number of entries to process (0 for all)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output for all entries"),
):
    """Process SFT JSONL data using AutoLinearTokenizer for full tokenization."""

    if verbose:
        print("SFT JSONL Data Processor with AutoLinearTokenizer")
        print("=" * 60)

        if no_tokenization:
            print("Tokenization disabled")

    # Fail fast - let any errors bubble up immediately
    load_and_process_jsonl(
        file_path=file_path,
        tokenize_text=not no_tokenization,
        max_entries=max_entries,
        verbose=verbose
    )


if __name__ == "__main__":
    typer.run(main)