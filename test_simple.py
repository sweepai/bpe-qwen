#!/usr/bin/env python3
"""
Simple test to debug tokenizer initialization.
"""

import bpe_qwen
import time

print("Testing bpe-qwen tokenizer initialization...")

start_time = time.time()
try:
    tokenizer = bpe_qwen.QwenTokenizer("vocab.json", "merges.txt")
    print(f"✓ Tokenizer loaded in {time.time() - start_time:.2f} seconds")
    print(f"Vocab size: {tokenizer.vocab_size()}")
    
    # Test simple encoding
    test_text = "Hello, world!"
    print(f"\nTesting encoding: '{test_text}'")
    tokens = tokenizer.encode(test_text)
    print(f"Tokens: {tokens}")
    
    # Test decoding
    print("\nTesting decoding...")
    decoded = tokenizer.decode(tokens)
    print(f"Decoded: '{decoded}'")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()