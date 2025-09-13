#!/usr/bin/env python3
"""Benchmark pretokenization performance: fast (two-pass) vs slow (fancy-regex)"""

import time
from bpe_qwen import bpe_qwen

# Test strings of various sizes
TEST_STRINGS = {
    "small": "Hello, world! This is a test string with some punctuation.",

    "medium": """The quick brown fox jumps over the lazy dog. This pangram sentence contains every letter 
    of the alphabet at least once. It's commonly used for testing fonts, keyboards, and now, 
    tokenizers! Let's add some more text to make it medium-sized. We can include numbers like 
    123456789 and special characters like @#$%^&*() to make it more interesting.""" * 3,

    "large": """Natural language processing (NLP) is a subfield of linguistics, computer science, 
    and artificial intelligence concerned with the interactions between computers and human language, 
    in particular how to program computers to process and analyze large amounts of natural language data. 
    The goal is a computer capable of "understanding" the contents of documents, including the contextual 
    nuances of the language within them. The technology can then accurately extract information and insights 
    contained in the documents as well as categorize and organize the documents themselves.

    Challenges in natural language processing frequently involve speech recognition, natural language understanding, 
    and natural language generation. Modern approaches to NLP are based on machine learning, especially deep learning. 
    Machine learning approaches have revolutionized NLP by enabling computers to learn from data without being 
    explicitly programmed.

    Deep learning models, particularly transformer-based architectures like BERT, GPT, and their variants, 
    have achieved state-of-the-art results on a wide range of NLP tasks. These models are pre-trained on 
    massive amounts of text data and can be fine-tuned for specific tasks with relatively small amounts of 
    task-specific training data. This approach, known as transfer learning, has dramatically reduced the amount 
    of labeled data required to achieve good performance on many NLP tasks.

    The applications of NLP are vast and growing. They include machine translation, sentiment analysis, 
    question answering, text summarization, named entity recognition, and many more. Virtual assistants like 
    Siri, Alexa, and Google Assistant rely heavily on NLP to understand and respond to user queries. 
    Social media platforms use NLP for content moderation and to understand user engagement. In healthcare, 
    NLP is used to extract information from clinical notes and medical literature. In finance, it's used for 
    analyzing news and reports to inform trading decisions.""" * 5,

    "huge": """
    def process_data(input_file, output_file, config=None):
        '''Process data from input file and write to output file.'''
        import json
        import pandas as pd
        from pathlib import Path

        # Default configuration
        default_config = {
            'chunk_size': 1000,
            'encoding': 'utf-8',
            'compression': 'gzip',
            'verbose': True
        }

        if config:
            default_config.update(config)

        # Read input file
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file {input_file} not found")

        # Process based on file extension
        if input_path.suffix == '.json':
            with open(input_path, 'r', encoding=default_config['encoding']) as f:
                data = json.load(f)
        elif input_path.suffix == '.csv':
            data = pd.read_csv(input_path, encoding=default_config['encoding'])
        else:
            with open(input_path, 'r', encoding=default_config['encoding']) as f:
                data = f.read()

        # Apply transformations
        if isinstance(data, pd.DataFrame):
            # Data cleaning
            data = data.dropna()
            data = data.drop_duplicates()

            # Feature engineering
            if 'timestamp' in data.columns:
                data['year'] = pd.to_datetime(data['timestamp']).dt.year
                data['month'] = pd.to_datetime(data['timestamp']).dt.month
                data['day'] = pd.to_datetime(data['timestamp']).dt.day

        # Write output
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == '.json':
            with open(output_path, 'w', encoding=default_config['encoding']) as f:
                json.dump(data, f, indent=2)
        elif output_path.suffix == '.csv':
            if isinstance(data, pd.DataFrame):
                data.to_csv(output_path, index=False, encoding=default_config['encoding'])
        else:
            with open(output_path, 'w', encoding=default_config['encoding']) as f:
                f.write(str(data))

        if default_config['verbose']:
            print(f"Processed {input_file} -> {output_file}")

        return data
    """ * 20  # Repeat to make it huge
}

# Test with special cases
TEST_STRINGS["whitespace_heavy"] = "   Hello    world   \n\n\n   How    are   you?   \t\t\tToday?   "
TEST_STRINGS["crlf"] = "line1\rline2\nline3\r\nline4\r\n\r\nmixed   spaces  and\tnewlines"

def benchmark_pretokenization():
    """Benchmark the exposed pretokenization functions"""

    print("=" * 80)
    print("PRETOKENIZATION BENCHMARK: Fast (two-pass) vs Slow (fancy-regex)")
    print("=" * 80)

    for name, text in TEST_STRINGS.items():
        print(f"\n{name.upper()} text ({len(text)} chars)")
        print("-" * 40)

        # Determine iteration count based on text size
        if len(text) < 100:
            iterations = 1000
        elif len(text) < 1000:
            iterations = 100
        elif len(text) < 10000:
            iterations = 10
        else:
            iterations = 3

        # Warm up both methods
        for _ in range(2):
            _ = bpe_qwen.pretokenize_slow(text)
            _ = bpe_qwen.pretokenize_fast(text)

        # Benchmark slow (fancy-regex with lookahead)
        start = time.perf_counter()
        for _ in range(iterations):
            slow_result = bpe_qwen.pretokenize_slow(text)
        slow_time = (time.perf_counter() - start) / iterations

        # Benchmark fast (two-pass with correction)
        start = time.perf_counter()
        for _ in range(iterations):
            fast_result = bpe_qwen.pretokenize_fast(text)
        fast_time = (time.perf_counter() - start) / iterations

        # Check correctness
        if slow_result == fast_result:
            correctness = "✓ MATCH"
        else:
            correctness = f"✗ MISMATCH ({len(slow_result)} vs {len(fast_result)} tokens)"

        # Calculate metrics
        speedup = slow_time / fast_time if fast_time > 0 else 0
        throughput_slow = len(text) / slow_time / 1_000_000 if slow_time > 0 else 0
        throughput_fast = len(text) / fast_time / 1_000_000 if fast_time > 0 else 0

        print(f"  Slow (fancy-regex):  {slow_time*1000:.3f} ms  ({throughput_slow:.2f} MB/s)")
        print(f"  Fast (two-pass):     {fast_time*1000:.3f} ms  ({throughput_fast:.2f} MB/s)")
        print(f"  Speedup:             {speedup:.2f}x")
        print(f"  Tokens produced:     {len(fast_result)}")
        print(f"  Correctness:         {correctness}")

if __name__ == "__main__":
    benchmark_pretokenization()