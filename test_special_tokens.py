import sys
sys.path.append('.')
from bpe_qwen import QwenTokenizer
from transformers import AutoTokenizer

# Load tokenizers
bpe_tokenizer = QwenTokenizer('data')
hf_tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-0.5B')

# Test special token handling
test_texts = [
    'Normal text with <|im_start|> special token',
    'text<|endoftext|>more',
    '<|im_start|>hello<|im_end|>',
    'prefix <|im_start|> middle <|im_end|> suffix',
]

for text in test_texts:
    print(f"\nTest: {repr(text)}")

    # Encode
    bpe_ids = bpe_tokenizer.encode(text)
    hf_ids = hf_tokenizer.encode(text, add_special_tokens=False)

    print(f"BPE IDs: {bpe_ids[:10]}..." if len(bpe_ids) > 10 else f"BPE IDs: {bpe_ids}")
    print(f"HF  IDs: {hf_ids[:10]}..." if len(hf_ids) > 10 else f"HF  IDs: {hf_ids}")

    # Decode
    bpe_decoded = bpe_tokenizer.decode(bpe_ids)
    hf_decoded = hf_tokenizer.decode(hf_ids, skip_special_tokens=False)

    print(f"BPE decoded: {repr(bpe_decoded)}")
    print(f"HF  decoded: {repr(hf_decoded)}")

    if bpe_decoded != hf_decoded:
        print(f"❌ MISMATCH!")
    else:
        print(f"✓ Match")

# Also test the special token IDs
print("\n\nSpecial token IDs:")
print(f"<|im_start|> ID in BPE: {bpe_tokenizer.encode('<|im_start|>')}")
print(f"<|im_start|> ID in HF: {hf_tokenizer.encode('<|im_start|>', add_special_tokens=False)}")
print(f"<|endoftext|> ID in BPE: {bpe_tokenizer.encode('<|endoftext|>')}")
print(f"<|endoftext|> ID in HF: {hf_tokenizer.encode('<|endoftext|>', add_special_tokens=False)}")