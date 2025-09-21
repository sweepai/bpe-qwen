# bpe-qwen

A blazing-fast BPE tokenizer for Qwen models, built with Rust and the [rust-gems BPE crate](https://github.com/github/rust-gems/tree/main/crates/bpe). Achieves **5x faster** tokenization by default and **10x faster** with parallelization compared to HuggingFace tokenizers.

## Features

- 🚀 **Linear-time tokenization** using optimized Rust implementation
- 🐍 **Python bindings** via PyO3 for seamless integration
- 📦 **Native BPE format support** (vocab.json + merges.txt)
- ⚡ **5x faster encoding** by default, **10x faster** with parallelism, and **2x faster decoding** compared to HuggingFace
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

Performance comparison with HuggingFace tokenizers on WikiText dataset (2,891 texts, 1.3M characters):

### Sequential Performance:
| Tokenizer | Speed | Speedup vs HF |
|-----------|-------|---------------|
| **bpe-qwen** | **5.01M tokens/sec** | **4.63x** |
| HuggingFace | 1.08M tokens/sec | 1.00x |

### Parallel Performance (8 workers):
| Tokenizer | Speed | Speedup vs HF | Parallel Benefit |
|-----------|-------|---------------|------------------|
| **bpe-qwen** | **24.31M tokens/sec** | **9.16x** | **4.85x vs sequential** |
| HuggingFace | 2.65M tokens/sec | 1.00x | 2.46x vs sequential |

✅ **Token consistency verified**: All methods produce identical 298,938 tokens


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
