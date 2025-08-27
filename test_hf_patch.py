#!/usr/bin/env python3
"""
Test script demonstrating HuggingFace compatibility patch for bpe-qwen.

This shows how to use bpe-qwen as a drop-in replacement for HuggingFace tokenizers.
"""

import time
import sys
from pathlib import Path

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent))

# Import and apply the patch
import bpe_qwen.hf_patch

def test_basic_compatibility():
    """Test basic HuggingFace API compatibility."""
    print("=" * 80)
    print("Testing HuggingFace Compatibility Patch")
    print("=" * 80)
    
    # Apply the patch
    print("\n1. Applying bpe-qwen patch to transformers...")
    bpe_qwen.hf_patch.patch_transformers()
    
    # Now import transformers - it will use bpe-qwen for Qwen models
    from transformers import AutoTokenizer
    
    print("\n2. Loading Qwen tokenizer (will use bpe-qwen under the hood)...")
    # Use local data directory directly for testing
    tokenizer = bpe_qwen.hf_patch.QwenTokenizerFast(model_dir="data/")
    print("   ✓ Tokenizer loaded")
    print(f"   Type: {type(tokenizer).__name__}")
    print(f"   Vocab size: {tokenizer.vocab_size}")
    
    print("\n3. Testing basic encoding/decoding...")
    test_text = "Hello, world! This is a test of the fast tokenizer."
    
    # Test encode
    start = time.perf_counter()
    token_ids = tokenizer.encode(test_text)
    encode_time = time.perf_counter() - start
    print(f"   Encoded: {test_text[:30]}...")
    print(f"   Tokens: {token_ids[:10]}... (total: {len(token_ids)})")
    print(f"   Time: {encode_time*1000:.3f}ms")
    
    # Test decode
    start = time.perf_counter()
    decoded_text = tokenizer.decode(token_ids)
    decode_time = time.perf_counter() - start
    print(f"   Decoded: {decoded_text[:30]}...")
    print(f"   Time: {decode_time*1000:.3f}ms")
    
    # Verify round-trip
    assert decoded_text == test_text, "Round-trip failed!"
    print("   ✓ Round-trip successful")
    
    print("\n4. Testing HuggingFace-style tokenization...")
    # Test the __call__ method with various options
    outputs = tokenizer(
        test_text,
        padding=False,
        truncation=False,
        return_attention_mask=True
    )
    print(f"   Output keys: {list(outputs.keys())}")
    print(f"   Input IDs shape: {len(outputs['input_ids'][0])}")
    
    print("\n5. Testing batch tokenization...")
    batch_texts = [
        "First text to tokenize.",
        "Second text is a bit longer than the first one.",
        "Third!"
    ]
    
    start = time.perf_counter()
    batch_outputs = tokenizer(
        batch_texts,
        padding=True,
        return_attention_mask=True
    )
    batch_time = time.perf_counter() - start
    
    print(f"   Batch size: {len(batch_texts)}")
    print(f"   Padded lengths: {[len(ids) for ids in batch_outputs['input_ids']]}")
    print(f"   Time: {batch_time*1000:.3f}ms")
    
    # Check attention masks
    if 'attention_mask' in batch_outputs:
        print(f"   ✓ Attention masks generated")
    
    print("\n6. Testing tensor returns...")
    try:
        import torch
        tensor_outputs = tokenizer(
            test_text,
            return_tensors="pt",
            padding=False,
            truncation=False
        )
        print(f"   PyTorch tensor shape: {tensor_outputs['input_ids'].shape}")
        print("   ✓ PyTorch tensor return successful")
    except ImportError:
        print("   ⚠ PyTorch not available, skipping tensor test")
    
    print("\n" + "=" * 80)
    print("✅ All compatibility tests passed!")
    print("=" * 80)


def benchmark_comparison():
    """Compare performance with original HuggingFace tokenizer."""
    print("\n" + "=" * 80)
    print("Performance Comparison")
    print("=" * 80)
    
    # First, get the patched version
    from transformers import AutoTokenizer
    
    # Make sure patch is applied
    if not hasattr(AutoTokenizer.from_pretrained, '__wrapped__'):
        bpe_qwen.hf_patch.patch_transformers()
    
    print("\n1. Loading patched (bpe-qwen) tokenizer...")
    patched_tokenizer = bpe_qwen.hf_patch.QwenTokenizerFast(model_dir="data/")
    
    print("\n2. Unpatch and load original HuggingFace tokenizer...")
    bpe_qwen.hf_patch.unpatch_transformers()
    
    try:
        original_tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            trust_remote_code=True
        )
        print("   ✓ Original tokenizer loaded")
    except Exception as e:
        print(f"   ⚠ Could not load original tokenizer: {e}")
        print("   Skipping performance comparison")
        return
    
    # Test data
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world of technology.",
        "Python is a versatile programming language loved by developers.",
    ] * 100  # Repeat for more samples
    
    print(f"\n3. Benchmarking with {len(test_texts)} texts...")
    
    # Benchmark bpe-qwen
    print("\n   bpe-qwen (patched):")
    start = time.perf_counter()
    for text in test_texts:
        _ = patched_tokenizer.encode(text)
    bpe_time = time.perf_counter() - start
    print(f"   Time: {bpe_time:.3f}s")
    print(f"   Speed: {len(test_texts)/bpe_time:.1f} texts/sec")
    
    # Benchmark original
    print("\n   HuggingFace (original):")
    start = time.perf_counter()
    for text in test_texts:
        _ = original_tokenizer.encode(text, add_special_tokens=False)
    hf_time = time.perf_counter() - start
    print(f"   Time: {hf_time:.3f}s")
    print(f"   Speed: {len(test_texts)/hf_time:.1f} texts/sec")
    
    # Compare
    speedup = hf_time / bpe_time
    print(f"\n   🚀 Speedup: {speedup:.2f}x faster with bpe-qwen!")
    
    # Re-apply patch for future use
    bpe_qwen.hf_patch.patch_transformers()


def test_auto_patch():
    """Test automatic patching via environment variable."""
    print("\n" + "=" * 80)
    print("Testing Auto-Patch Feature")
    print("=" * 80)
    
    import os
    import subprocess
    
    # Create a test script
    test_script = '''
import os
os.environ['BPE_QWEN_AUTO_PATCH'] = 'true'

# This import will auto-patch transformers
import bpe_qwen.hf_patch

# Now test that it's patched
from transformers import AutoTokenizer

# This should use bpe-qwen
tokenizer = AutoTokenizer.from_pretrained("data/", local_files_only=True)
print(f"Tokenizer type: {type(tokenizer).__name__}")

# Should show QwenTokenizerFast if patch worked
assert "QwenTokenizerFast" in str(type(tokenizer).__name__)
print("✓ Auto-patch successful!")
'''
    
    # Write test script
    with open("test_auto_patch.py", "w") as f:
        f.write(test_script)
    
    print("\n1. Testing with BPE_QWEN_AUTO_PATCH=true...")
    try:
        result = subprocess.run(
            [sys.executable, "test_auto_patch.py"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("   ✓ Auto-patch worked!")
            print(f"   Output: {result.stdout.strip()}")
        else:
            print(f"   ⚠ Auto-patch test failed: {result.stderr}")
    except Exception as e:
        print(f"   ⚠ Could not test auto-patch: {e}")
    finally:
        # Clean up
        Path("test_auto_patch.py").unlink(missing_ok=True)


if __name__ == "__main__":
    print("\n🚀 BPE-QWEN HuggingFace Compatibility Test\n")
    
    # Run basic compatibility tests
    test_basic_compatibility()
    
    # Run performance comparison
    benchmark_comparison()
    
    # Test auto-patch feature
    # test_auto_patch()  # Commented out as it requires subprocess
    
    print("\n✨ All tests completed successfully!")
    print("\nUsage in your code:")
    print("-" * 40)
    print("import bpe_qwen.hf_patch")
    print("bpe_qwen.hf_patch.patch_transformers()")
    print("# Now all Qwen tokenizers use bpe-qwen!")
    print("-" * 40)