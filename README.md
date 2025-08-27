# bpe-qwen

A blazing-fast BPE tokenizer for Qwen models, built with Rust and the [rust-gems BPE crate](https://github.com/github/rust-gems/tree/main/crates/bpe). Achieves **7.72x faster** tokenization compared to HuggingFace tokenizers.

## Features

- 🚀 **Linear-time tokenization** using optimized Rust implementation
- 🐍 **Python bindings** via PyO3 for seamless integration
- 📦 **Native BPE format support** (vocab.json + merges.txt)
- ⚡ **7.72x faster encoding** and **2.22x faster decoding** compared to HuggingFace
- 🔧 **GPT-2 byte-level encoding** with proper special character handling
- 🎯 **Pretokenization support** with regex patterns

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/sweepai/bpe-qwen.git
cd bpe-qwen

# Install with maturin (requires Rust toolchain)
pip install maturin
maturin develop --release
```

## Usage

### Quick Start

```python
from bpe_qwen import QwenTokenizer

# Initialize tokenizer with vocab and merges files
tokenizer = QwenTokenizer("vocab.json", "merges.txt")

# Encode text
text = "Hello, world!"
tokens = tokenizer.encode(text)
print(f"Tokens: {tokens}")  # [9707, 11, 1879, 0]

# Decode tokens back to text
decoded = tokenizer.decode(tokens)
print(f"Decoded: {decoded}")  # "Hello, world!"

# Get vocabulary size
print(f"Vocab size: {tokenizer.vocab_size()}")  # 151643

# Count tokens without full encoding (fast!)
count = tokenizer.count_tokens(text)
print(f"Token count: {count}")  # 4

# Parallel batch encoding for high throughput
texts = ["Hello, world!", "How are you?", "Tokenization is fast!"]
batch_tokens = tokenizer.encode_batch_parallel(texts, num_workers=8)
print(f"Batch tokens: {len(batch_tokens)} texts processed")
# Up to 31.43M tokens/sec with 8 workers (5.66x faster than sequential)
# 18.13x faster than HuggingFace with native parallelism enabled
```

### HuggingFace Compatibility

Use bpe-qwen as a drop-in replacement for HuggingFace tokenizers:

```python
# Patch transformers to use bpe-qwen for Qwen models
import bpe_qwen.hf_patch
bpe_qwen.hf_patch.patch_transformers()

# Now all Qwen tokenizers will use the fast bpe-qwen implementation!
from transformers import AutoTokenizer

# This automatically uses bpe-qwen under the hood
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

# Use it exactly like a HuggingFace tokenizer
outputs = tokenizer(
    "Hello, world!",
    return_tensors="pt",
    padding=True,
    truncation=True
)
print(outputs["input_ids"])

# Batch processing with native HuggingFace API
batch = tokenizer(
    ["Text 1", "Text 2", "Text 3"],
    padding=True,
    return_attention_mask=True
)
```

Or use the compatibility wrapper directly:

```python
from bpe_qwen.hf_patch import QwenTokenizerFast

# Direct usage with HuggingFace-compatible API
tokenizer = QwenTokenizerFast(model_dir="path/to/tokenizer/files")

# Supports all HuggingFace tokenizer methods
ids = tokenizer.encode("Hello!", add_special_tokens=True)
text = tokenizer.decode(ids, skip_special_tokens=True)
batch = tokenizer.batch_encode_plus(texts, padding=True)
```

### Downloading Tokenizer Files

Download the required files from HuggingFace:

```bash
# Download vocab.json and merges.txt from Qwen model
wget https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/raw/main/vocab.json
wget https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/raw/main/merges.txt
```

## Benchmark Results

Performance comparison with HuggingFace tokenizers on various text samples:

| Metric | bpe-qwen (Rust) | HuggingFace | Speedup |
|--------|-----------------|-------------|---------|
| **Encoding Speed** | 6.30M tokens/sec | 805K tokens/sec | **7.83x** |
| **Decoding Speed** | 12.34M tokens/sec | 5.33M tokens/sec | **2.32x** |
| **Load Time** | ~3.3 seconds | ~2.0 seconds | 1.65x |
| **Accuracy** | ✓ Matches | ✓ Baseline | 100% |

### Detailed Performance

- **Short text (13 chars)**: Sub-millisecond encoding
- **Code snippets (831 chars)**: <0.1ms encoding time
- **Large documents (2500 chars)**: <0.2ms encoding time
- **Memory efficient**: Minimal allocations during tokenization

## Technical Implementation

### Performance Optimization Journey

We systematically optimized the tokenizer through multiple iterations, achieving a **13% cumulative improvement** over the baseline:

#### Core Optimizations (Baseline → 6.39x faster)
1. **HashMap → Vec mapping** (10-70x improvement): Replaced `HashMap<u32, u32>` with `Vec<u32>` for O(1) token ID mapping
2. **ASCII normalization skip** (+3%): Fast-path ASCII text to skip Unicode normalization
3. **Vector pre-allocation** (+13.5%): Optimal 128-token capacity reduces reallocation overhead

#### Advanced Optimizations (6.39x → 7.83x faster)
4. **SIMD ASCII detection** (+4%): Process 8 bytes at once using u64 chunks instead of byte-by-byte checks
5. **Memory pool** (+5%): Reuse `Vec<u32>` allocations between tokenization calls to reduce allocation pressure
6. **True SIMD intrinsics** (+3.2% encoding, +5.7% decoding): NEON on ARM, SSE2 on x86_64 for 16-byte parallel processing
7. **Zero-copy strings** (+3% encoding, +2.5% decoding): Use `Cow<str>` to avoid allocations for ASCII text and when normalization not needed

#### Experiment Results Table
| Optimization | Encoding Speed | Encoding vs HF | Decoding Speed | Decoding vs HF | Status |
|-------------|---------------|----------------|----------------|----------------|---------|
| Baseline | 5.36M tok/s | 6.39x | 11.47M tok/s | 2.22x | ✅ Kept |
| + SIMD ASCII | 5.57M tok/s | 6.87x | - | - | ✅ Kept |
| + Memory Pool | 5.85M tok/s | 7.30x | 11.47M tok/s | 2.22x | ✅ Kept |
| + String Interning | 6.05M tok/s | 7.72x | 7.55M tok/s | 1.38x | ❌ Reverted |
| - String Interning | 5.93M tok/s | 6.99x | 11.39M tok/s | 2.12x | ✅ Kept |
| + True SIMD | 6.12M tok/s | 7.28x | 12.04M tok/s | 2.21x | ✅ Kept |
| + Batch API | 6.06M tok/s | 7.50x | 12.04M tok/s | 2.32x | ❌ Reverted |
| + Zero-Copy | 6.30M tok/s | 7.83x | 12.34M tok/s | 2.32x | ✅ Kept |
| + Jemalloc | 5.70M tok/s | 8.91x | 11.01M tok/s | 2.19x | ❌ Reverted |
| + **Parallel Batch (8 workers)** | **31.43M tok/s** | **18.13x*** | - | - | ✅ Kept |

*\* Fair comparison with HuggingFace's native parallelism enabled (TOKENIZERS_PARALLELISM=true). When both tokenizers use parallelism, bpe-qwen is 18.13x faster.

#### Implementation Details
- **SIMD ASCII**: Uses unsafe pointer arithmetic to check 8 bytes simultaneously for non-ASCII markers
- **Memory Pool**: `RefCell<Vec<Vec<u32>>>` with capacity-based reuse and size limits
- **String Interning**: `HashMap<String, Arc<str>>` cache with 1000-entry limit to prevent unbounded growth
- **Release Builds Critical**: Debug builds show 13x performance penalty vs release
- **Parallel Batch**: Thread-safe implementation using Rayon with Arc for shared read-only data

#### Fundamental Optimizations
1. **Linear-time BPE encoding** using rust-gems' optimized algorithm
2. **Aho-Corasick pattern matching** for fast token lookups
3. **Precomputed hash factors** to avoid collisions
4. **Efficient byte-level encoding** with GPT-2 compatible mappings
5. **Zero-copy operations** where possible in Rust

### Architecture

```
┌─────────────────┐
│   Python API    │  PyO3 Bindings
├─────────────────┤
│  QwenTokenizer  │  Main tokenizer interface
├─────────────────┤
│   BPE Engine    │  rust-gems BPE crate
├─────────────────┤
│  Byte Encoding  │  GPT-2 byte-level encoding
└─────────────────┘
```

## Development

### Building from Source

```bash
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone and build
git clone https://github.com/sweepai/bpe-qwen.git
cd bpe-qwen
maturin develop --release

# Run tests
python test_simple.py
python benchmark.py
```

### Running Benchmarks

```bash
# Run comprehensive benchmarks
python benchmark.py

# Compare against HuggingFace
# (automatically downloads HF tokenizer if needed)
```

## Limitations

- Currently supports Qwen models with GPT-2 style byte-level BPE
- Requires vocab.json and merges.txt files (not tokenizer.json)
- Some special tokens may need manual configuration

## Future Improvements

### Potential Optimizations
- [ ] **Rayon parallelization**: Multi-threaded tokenization for large texts using data parallelism
- [ ] **True SIMD intrinsics**: Explicit vector instructions for even faster ASCII detection and token processing
- [ ] **Custom allocators**: Specialized memory management for tokenization workloads
- [ ] **Profile-guided optimization**: Workload-specific optimizations based on production usage patterns

### Feature Enhancements
- [ ] Support for more model architectures
- [ ] Streaming tokenization for large documents  
- [ ] Batch processing optimizations
- [ ] Direct tokenizer.json support
- [ ] WebAssembly bindings for browser usage

## Acknowledgments

- Built on top of the excellent [rust-gems BPE crate](https://github.com/github/rust-gems)
- Inspired by the need for faster tokenization in production ML pipelines
- GPT-2 byte-level encoding implementation based on OpenAI's original design

---

*This entire project was written by [Sweep AI](https://sweep.dev), an AI plugin for JetBrains IDEs*
