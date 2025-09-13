#!/usr/bin/env python3
"""Debug script to understand emoji tokenization issues"""

import bpe_qwen

# The failing emoji test case
test_text = 'family: 👨\u200d👩\u200d👧\u200d👦 and flags: 🇺🇸 🇨🇦'

print(f"Test text: {test_text}")
print(f"Test repr: {repr(test_text)}")
print()

# Create tokenizer
tokenizer = bpe_qwen.QwenTokenizer("/Users/kevinlu/next-edit/bpe-qwen/data")

# Try encoding and decoding
try:
    tokens = tokenizer.encode(test_text)
    print(f"Tokens: {tokens}")

    decoded = tokenizer.decode(tokens)
    print(f"Decoded: {decoded}")
    print(f"Decoded repr: {repr(decoded)}")

    if decoded != test_text:
        print(f"\nMISMATCH!")
        print(f"Original: {repr(test_text)}")
        print(f"Decoded:  {repr(decoded)}")

        # Show character-by-character comparison
        print("\nCharacter comparison:")
        for i, (orig, dec) in enumerate(zip(test_text, decoded)):
            if orig != dec:
                print(f"  Position {i}: orig={repr(orig)} (U+{ord(orig):04X}) vs dec={repr(dec)} (U+{ord(dec):04X})")

except Exception as e:
    print(f"Error: {e}")

# Also test with HuggingFace tokenizer for comparison
print("\n" + "="*50)
print("HuggingFace Tokenizer comparison:")
try:
    from transformers import AutoTokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)

    hf_tokens = hf_tokenizer.encode(test_text)
    print(f"HF Tokens: {hf_tokens}")

    hf_decoded = hf_tokenizer.decode(hf_tokens)
    print(f"HF Decoded: {hf_decoded}")
    print(f"HF Decoded repr: {repr(hf_decoded)}")

    if hf_decoded == test_text:
        print("HuggingFace handles it correctly!")
    else:
        print("HuggingFace also has issues with this text")

except Exception as e:
    print(f"Error loading HF tokenizer: {e}")