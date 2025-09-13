#!/usr/bin/env python3
"""
Test to verify that bpe-qwen produces identical results to HuggingFace tokenizer.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bpe_qwen
import json
import difflib
from typing import List, Tuple

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

def load_test_texts() -> List[str]:
    """Load various test texts to verify."""
    return [
        # Basic tests
        "Hello, world!",
        "",
        " ",
        "   ",
        "\n",
        "\n\n",
        "\t",
        
        # Unicode and special characters  
        "你好世界",
        "مرحبا بالعالم",
        "Здравствуй мир",
        "こんにちは世界",
        "🚀🔥💻",
        "émojis and àccénts",
        
        # Code snippets
        "def hello_world():\n    print('Hello, World!')",
        "const x = 42; // comment",
        "#include <iostream>",
        "SELECT * FROM users WHERE id = 1;",
        
        # Edge cases
        "a" * 1000,  # Long repetitive text
        "The quick brown fox jumps over the lazy dog. " * 10,
        "1234567890" * 100,
        "\n".join(["Line " + str(i) for i in range(100)]),
        
        # Mixed content
        """
        This is a test with multiple lines.
        It includes code: x = lambda y: y * 2
        And unicode: 你好 мир العالم
        And numbers: 3.14159, 2.71828
        """,
        
        # Special tokens that might exist
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "Normal text with <|im_start|> special token",
        
        # Whitespace variations
        "word1  word2   word3    word4",
        "\r\n",
        "line1\rline2\nline3\r\nline4",
        
        # Real text samples
        """The WikiText language modeling dataset is a collection of over 100 million tokens 
        extracted from the set of verified Good and Featured articles on Wikipedia.""",
        
        """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)""",
        
        # Empty and whitespace edge cases
        "",
        " " * 100,
        "\n" * 50,
        "\t" * 20,
    ]

def verify_tokenization(qwen_tokenizer, hf_tokenizer, text: str) -> Tuple[bool, str, List[int], List[int], float]:
    """
    Verify that both tokenizers produce identical results.
    Returns: (match, error_msg, qwen_tokens, hf_tokens, diff_percentage)
    """
    try:
        # Encode with our tokenizer
        qwen_tokens = qwen_tokenizer.encode(text)
        
        # Encode with HuggingFace
        hf_tokens = hf_tokenizer(text, return_tensors=None, add_special_tokens=False)['input_ids']
        
        # Calculate difference percentage using difflib
        def calculate_diff_percentage(tokens1, tokens2):
            if len(tokens1) == 0 and len(tokens2) == 0:
                return 0.0

            # Use difflib.SequenceMatcher to calculate similarity ratio
            matcher = difflib.SequenceMatcher(None, tokens1, tokens2)
            similarity_ratio = matcher.ratio()

            # Convert similarity to difference percentage
            difference_percentage = (1.0 - similarity_ratio) * 100.0
            return difference_percentage

        diff_percentage = calculate_diff_percentage(qwen_tokens, hf_tokens)

        # Check if tokens match
        if qwen_tokens == hf_tokens:
            # Also verify decoding
            qwen_decoded = qwen_tokenizer.decode(qwen_tokens)
            hf_decoded = hf_tokenizer.decode(hf_tokens, skip_special_tokens=False)
            
            # For empty or whitespace-only text, both might normalize differently
            if text.strip() == "":
                return True, "", qwen_tokens, hf_tokens, diff_percentage

            # Check if decoded text matches (allowing for some normalization differences)
            if qwen_decoded == hf_decoded or qwen_decoded.strip() == hf_decoded.strip():
                return True, "", qwen_tokens, hf_tokens, diff_percentage
            else:
                return False, f"Decoded text mismatch: '{qwen_decoded}' vs '{hf_decoded}'", qwen_tokens, hf_tokens, diff_percentage
        else:
            return False, f"Token mismatch: {qwen_tokens[:10]}... vs {hf_tokens[:10]}... (diff: {diff_percentage:.1f}%)", qwen_tokens, hf_tokens, diff_percentage
            
    except Exception as e:
        return False, f"Exception: {str(e)}", [], [], 0.0

def main():
    """Main test function."""
    print("="*80)
    print("Verification Test: bpe-qwen vs HuggingFace Tokenizer")
    print("="*80)
    
    # Load tokenizers
    print("\nLoading tokenizers...")
    
    # Load our tokenizer
    try:
        qwen_tokenizer = bpe_qwen.QwenTokenizer("data")
        print(f"✓ bpe-qwen loaded, vocab size: {qwen_tokenizer.vocab_size()}")
    except Exception as e:
        print(f"✗ Failed to load bpe-qwen: {e}")
        return 1
    
    # Load HuggingFace tokenizer
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
            return 1
    
    # Load test texts
    test_texts = load_test_texts()
    print(f"\nTesting {len(test_texts)} text samples...")
    
    # Track results
    passed = 0
    failed = 0
    failed_cases = []
    total_diff_percentage = 0.0

    # Test each text
    for i, text in enumerate(test_texts):
        # Show progress
        if i % 10 == 0:
            print(f"  Testing {i}/{len(test_texts)}...")
        
        # Verify tokenization
        match, error_msg, qwen_tokens, hf_tokens, diff_percentage = verify_tokenization(qwen_tokenizer, hf_tokenizer, text)

        # Add to total difference percentage
        total_diff_percentage += diff_percentage

        if match:
            passed += 1
        else:
            failed += 1
            preview = repr(text[:50] + "..." if len(text) > 50 else text)
            failed_cases.append({
                "text": text,
                "preview": preview,
                "error": error_msg,
                "qwen_tokens": qwen_tokens[:20],  # First 20 tokens
                "hf_tokens": hf_tokens[:20],
                "diff_percentage": diff_percentage,
            })
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Passed: {passed}/{len(test_texts)} ({passed/len(test_texts)*100:.1f}%)")
    print(f"Failed: {failed}/{len(test_texts)}")

    # Calculate and display average difference percentage
    average_diff_percentage = total_diff_percentage / len(test_texts)
    print(f"Average difference: {average_diff_percentage:.2f}%")

    if failed > 0:
        print("\nFailed cases:")
        for i, case in enumerate(failed_cases[:10], 1):  # Show first 10 failures
            print(f"\n{i}. Text: {case['preview']}")
            print(f"   Error: {case['error']}")
            print(f"   Difference: {case['diff_percentage']:.1f}%")
            hf_tokens = [hf_tokenizer.decode([token]) for token in case['hf_tokens']]
            print(f"   HuggingFace decoded: {hf_tokens}")
            qwen_tokens = [qwen_tokenizer.decode([token]) for token in case['qwen_tokens']]
            print(f"   bpe-qwen decoded: {qwen_tokens}")
    
    # Test performance on larger text
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON ON LARGE TEXT")
    print("="*80)
    
    # Create a large text
    large_text = " ".join(test_texts) * 10
    print(f"Large text size: {len(large_text):,} characters")
    
    import time
    
    # Benchmark our tokenizer
    start = time.perf_counter()
    qwen_tokens = qwen_tokenizer.encode(large_text)
    qwen_time = time.perf_counter() - start
    
    # Benchmark HuggingFace
    start = time.perf_counter()
    hf_tokens = hf_tokenizer(large_text, return_tensors=None, add_special_tokens=False)['input_ids']
    hf_time = time.perf_counter() - start
    
    print(f"\nbpe-qwen:")
    print(f"  Time: {qwen_time*1000:.2f} ms")
    print(f"  Tokens: {len(qwen_tokens):,}")
    print(f"  Speed: {len(large_text)/qwen_time:,.0f} chars/sec")
    
    print(f"\nHuggingFace:")
    print(f"  Time: {hf_time*1000:.2f} ms")
    print(f"  Tokens: {len(hf_tokens):,}")
    print(f"  Speed: {len(large_text)/hf_time:,.0f} chars/sec")
    
    print(f"\nSpeedup: {hf_time/qwen_time:.2f}x")
    
    # Overall verdict
    print("\n" + "="*80)
    if failed == 0:
        print("✅ VERIFICATION PASSED: All tokenizations match exactly!")
    else:
        print(f"⚠️  VERIFICATION FAILED: {failed} mismatches found")
        
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit(main())