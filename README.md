# bpe-qwen

A blazing-fast BPE tokenizer for Qwen models, built with Rust and the [rust-gems BPE crate](https://github.com/github/rust-gems/tree/main/crates/bpe). Achieves **~5x faster** tokenization compared to HuggingFace tokenizers.

## Features

- 🚀 **Linear-time tokenization** using optimized Rust implementation
- 🐍 **Python bindings** via PyO3 for seamless integration
- 📦 **Native BPE format support** (vocab.json + merges.txt)
- ⚡ **~5x faster encoding/decoding** compared to HuggingFace
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
| **Encoding Speed** | 2.57M tokens/sec | 547K tokens/sec | **4.69x** |
| **Decoding Speed** | 22.4M tokens/sec | 4.94M tokens/sec | **4.53x** |
| **Load Time** | ~3.3 seconds | ~6 seconds | ~1.8x |
| **Accuracy** | ✓ Matches | ✓ Baseline | 100% |

### Detailed Performance

- **Short text (13 chars)**: Sub-millisecond encoding
- **Code snippets (831 chars)**: <0.1ms encoding time
- **Large documents (2500 chars)**: <0.2ms encoding time
- **Memory efficient**: Minimal allocations during tokenization

## Technical Implementation

### Key Optimizations

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

- [ ] Support for more model architectures
- [ ] Streaming tokenization for large documents
- [ ] Batch processing optimizations
- [ ] Direct tokenizer.json support
- [ ] WebAssembly bindings for browser usage

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Built on top of the excellent [rust-gems BPE crate](https://github.com/github/rust-gems)
- Inspired by the need for faster tokenization in production ML pipelines
- GPT-2 byte-level encoding implementation based on OpenAI's original design

---

*Built by [Sweep AI](https://sweep.dev), an AI plugin for JetBrains IDEs*