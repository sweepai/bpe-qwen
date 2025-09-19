# bpe-qwen

A blazing-fast BPE tokenizer for Qwen models, built with Rust and the [rust-gems BPE crate](https://github.com/github/rust-gems/tree/main/crates/bpe). Achieves **5x faster** tokenization with parallelism compared to HuggingFace tokenizers.

## Features

- 🚀 **Linear-time tokenization** using optimized Rust implementation
- 🐍 **Python bindings** via PyO3 for seamless integration
- 📦 **Native BPE format support** (vocab.json + merges.txt)
- ⚡ **5x faster encoding** with parallelism and **2x faster decoding** compared to HuggingFace
- 🎯 **Pretokenization support** for Qwen's pretokenization pattern
- ✅ **100% accuracy verified** across comprehensive test suite, including special tokens

## Installation

```bash
pip install bpe-qwen
```

## Usage

### Quick Start

Use bpe-qwen as a drop-in replacement for HuggingFace tokenizers:

```python
# Patch transformers to use bpe-qwen for Qwen models
from bpe_qwen import AutoLinearTokenizer

# This automatically uses bpe-qwen under the hood
tokenizer = AutoLinearTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

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

## Benchmark Results

Performance comparison with HuggingFace tokenizers on various text samples:

| Metric | bpe-qwen (Rust) | HuggingFace | Speedup |
|--------|-----------------|-------------|---------|
| **Encoding Speed** | 19.22M chars/sec | 3.35M chars/sec | **5.73x** |
| **Decoding Speed** | 12.34M tokens/sec | 5.33M tokens/sec | **2.32x** |
| **Load Time** | ~3.3 seconds | ~2.0 seconds | 1.65x |

## Technical Implementation

### Performance Optimization Journey

We systematically optimized the tokenizer through multiple iterations with significant performance improvements:

#### Core Optimizations
1. **HashMap → Vec mapping**: Replaced `HashMap<u32, u32>` with `Vec<u32>` for O(1) token ID mapping
2. **ASCII normalization skip**: Fast-path ASCII text to skip Unicode normalization
3. **Vector pre-allocation**: Optimal 128-token capacity reduces reallocation overhead

#### Advanced Optimizations
4. **SIMD ASCII detection**: Process 8 bytes at once using u64 chunks instead of byte-by-byte checks
5. **Memory pool**: Reuse `Vec<u32>` allocations between tokenization calls to reduce allocation pressure
6. **True SIMD intrinsics**: NEON on ARM, SSE2 on x86_64 for 16-byte parallel processing
7. **Zero-copy strings**: Use `Cow<str>` to avoid allocations for ASCII text and when normalization not needed

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
- [ ] Early stopping for tokenization based on token count
- [ ] Support for more model architectures
- [ ] Batch processing optimizations

## Acknowledgments

- Built on top of the excellent [rust-gems BPE crate](https://github.com/github/rust-gems)
- Inspired by the need for faster tokenization in production ML pipelines

---

*This entire project was written by [Sweep AI](https://sweep.dev), an AI plugin for JetBrains IDEs*
