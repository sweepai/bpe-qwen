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

    # More robust decoded text comparison
    if auto_decoded == hf_decoded:
        decoded_match = True
    elif auto_decoded.strip() == hf_decoded.strip():
        decoded_match = True
    elif text.strip() == "" and auto_decoded.strip() == hf_decoded.strip():
        # For empty or whitespace-only text, both might normalize differently
        decoded_match = True
    else:
        decoded_match = False

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
            'has_mismatch': not tokens_match,
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
    print(f"{name1:>8}: {context1}")

    # Show context for second list
    context2 = list2[start:end2]
    print(f"{name2:>8}: {context2}")

    # Show pointer to mismatch position
    pointer_pos = mismatch_pos - start
    if pointer_pos < len(context1) and pointer_pos < len(context2):
        spaces = " " * (11 + sum(len(repr(token)) + 2 for token in context1[:pointer_pos]))
        print(f"{spaces}^ mismatch here")


def show_token_mismatch_with_decoded(auto_token_ids, hf_token_ids, auto_tokenizer, hf_tokenizer, context_size=5):
    """
    Show context around the first mismatch between two token lists with both token IDs and decoded strings.

    Args:
        auto_token_ids: AutoLinear token IDs
        hf_token_ids: HuggingFace token IDs
        auto_tokenizer: AutoLinear tokenizer instance
        hf_tokenizer: HuggingFace tokenizer instance
        context_size: Number of tokens to show before and after mismatch
    """
    mismatch_pos = find_first_mismatch_position(auto_token_ids, hf_token_ids)
    if mismatch_pos == -1:
        print("   No token ID mismatch found")
        return

    print(f"\n🔍 First token mismatch at position {mismatch_pos}:")

    # Calculate context window
    start = max(0, mismatch_pos - context_size)
    auto_end = min(len(auto_token_ids), mismatch_pos + context_size + 1)
    hf_end = min(len(hf_token_ids), mismatch_pos + context_size + 1)

    # Get context token IDs
    auto_context_ids = auto_token_ids[start:auto_end]
    hf_context_ids = hf_token_ids[start:hf_end]

    # Decode individual tokens
    auto_context_decoded = []
    hf_context_decoded = []

    for token_id in auto_context_ids:
        try:
            decoded = auto_tokenizer.decode([token_id])
            auto_context_decoded.append(decoded)
        except:
            auto_context_decoded.append(f"<ERROR:{token_id}>")

    for token_id in hf_context_ids:
        try:
            decoded = hf_tokenizer.decode([token_id])
            hf_context_decoded.append(decoded)
        except:
            hf_context_decoded.append(f"<ERROR:{token_id}>")

    # Show token IDs with position markers
    print(f"   Token positions: {list(range(start, start + len(auto_context_ids)))}")
    print(f"   AutoLinear IDs:  {auto_context_ids}")
    print(f"   HuggingFace IDs: {hf_context_ids}")

    # Show decoded tokens
    print(f"   AutoLinear decoded:  {[repr(token) for token in auto_context_decoded]}")
    print(f"   HuggingFace decoded: {[repr(token) for token in hf_context_decoded]}")

    # Show pointer to mismatch position
    pointer_pos = mismatch_pos - start
    if pointer_pos < len(auto_context_ids) and pointer_pos < len(hf_context_ids):
        print(f"   Mismatch at position {mismatch_pos}: {auto_context_ids[pointer_pos]} vs {hf_context_ids[pointer_pos]}")
        print(f"   Decoded mismatch: {repr(auto_context_decoded[pointer_pos])} vs {repr(hf_context_decoded[pointer_pos])}")

    # Calculate and show mismatch percentage
    min_len = min(len(auto_token_ids), len(hf_token_ids))
    max_len = max(len(auto_token_ids), len(hf_token_ids))

    # Count mismatches in the overlapping portion
    mismatches = 0
    for i in range(min_len):
        if auto_token_ids[i] != hf_token_ids[i]:
            mismatches += 1

    # Add length difference as mismatches
    length_diff = abs(len(auto_token_ids) - len(hf_token_ids))
    total_mismatches = mismatches + length_diff

    # Calculate percentage based on the longer sequence
    mismatch_percentage = (total_mismatches / max_len) * 100 if max_len > 0 else 0

    print(f"   Mismatch percentage: {mismatch_percentage:.2f}% ({total_mismatches}/{max_len} tokens)")


def analyze_tokenization_results(results, text_name="text", verbose=False, auto_tokenizer=None, hf_tokenizer=None):
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
                if auto_tokenizer and hf_tokenizer:
                    show_token_mismatch_with_decoded(auto_results['token_ids'], hf_results['token_ids'],
                                                   auto_tokenizer, hf_tokenizer, context_size=5)
                else:
                    show_mismatch_context(auto_results['token_ids'], hf_results['token_ids'],
                                        "AutoLinear", "HuggingFace")
            else:
                # Token IDs are the same, but show them for debugging decoding issues
                print("   - Token IDs are identical")
                print(f"   Sample token IDs: {auto_results['token_ids'][:10]}{'...' if len(auto_results['token_ids']) > 10 else ''}")
                print(f"   Last token IDs: {auto_results['token_ids'][-10:] if len(auto_results['token_ids']) > 10 else auto_results['token_ids']}")

            if not comparison['decoded_match']:
                print("   - Decoded text differs")
                print(f"   AutoLinear decoded: {repr(auto_results['decoded_text'][:100])}")
                print(f"   HuggingFace decoded: {repr(hf_results['decoded_text'][:100])}")
                print(f"   AutoLinear length: {len(auto_results['decoded_text'])}")
                print(f"   HuggingFace length: {len(hf_results['decoded_text'])}")
                # Show character-by-character diff for debugging
                if len(auto_results['decoded_text']) != len(hf_results['decoded_text']):
                    print("   Length mismatch detected!")
                else:
                    for i, (a, h) in enumerate(zip(auto_results['decoded_text'], hf_results['decoded_text'])):
                        if a != h:
                            print(f"   First char diff at position {i}: {repr(a)} vs {repr(h)}")
                            break
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
                    prompt_analysis = analyze_tokenization_results(prompt_results, f"prompt {processed_entries + 1}", verbose=verbose, auto_tokenizer=auto_tokenizer, hf_tokenizer=hf_tokenizer)

                    # Compare tokenization for completion - fail fast on any tokenization errors
                    completion_results = compare_tokenization_methods(auto_tokenizer, hf_tokenizer, completion, f"completion {processed_entries + 1}")
                    completion_analysis = analyze_tokenization_results(completion_results, f"completion {processed_entries + 1}", verbose=verbose, auto_tokenizer=auto_tokenizer, hf_tokenizer=hf_tokenizer)

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