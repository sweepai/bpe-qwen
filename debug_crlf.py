#!/usr/bin/env python3
"""Debug script to understand how \r\n is being tokenized"""

import regex as re

# The patterns
QWEN_PATTERN_WITHOUT_LOOKAHEAD = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+"

def test_patterns():
    text = "line1\rline2\nline3\r\nline4"

    print("Testing text:", repr(text))
    print()

    # Test standard regex (our current behavior)
    print("Standard regex without lookahead:")
    std_re = re.compile(QWEN_PATTERN_WITHOUT_LOOKAHEAD)
    matches = list(std_re.finditer(text))
    for i, m in enumerate(matches):
        print(f"  Match {i}: {repr(m.group())} at position {m.start()}-{m.end()}")

    print()

    # Test just the \r\n part
    print("Testing just \\r\\n matching:")
    test_text = "\r\n"

    # Pattern that should match \r\n together
    pattern = r"\s*[\r\n]+"
    print(f"  Pattern: {pattern}")

    std_re = re.compile(pattern)
    matches = list(std_re.finditer(test_text))
    print(f"  Standard regex matches for {repr(test_text)}:")
    for m in matches:
        print(f"    {repr(m.group())}")

    # Try different test cases
    print()
    print("Testing pattern parts individually:")

    patterns_to_test = [
        (r"[^\r\n\p{L}\p{N}]?\p{L}+", "letters with optional prefix"),
        (r"\p{N}", "numbers"),
        (r" ?[^\s\p{L}\p{N}]+[\r\n]*", "punctuation with optional space prefix and newlines suffix"),
        (r"\s*[\r\n]+", "newlines with optional whitespace prefix"),
        (r"\s+", "whitespace"),
    ]

    for pat, desc in patterns_to_test:
        print(f"\n  Pattern: {pat} ({desc})")
        try:
            r = re.compile(pat)
            for test in ["line3", "\r", "\n", "\r\n", " ", "  ", "line3\r\n"]:
                matches = list(r.finditer(test))
                if matches:
                    print(f"    {repr(test)} -> {[repr(m.group()) for m in matches]}")
        except Exception as e:
            print(f"    Error: {e}")

    # Test the order of alternatives
    print("\n\nTesting order of alternatives in full pattern:")
    text = "line3\r\nline4"
    matches = list(re.compile(QWEN_PATTERN_WITHOUT_LOOKAHEAD).finditer(text))
    for m in matches:
        print(f"  {repr(m.group())} at {m.start()}-{m.end()}")

    # Test each pattern component separately against \r\n
    print("\n\nTesting which part of the pattern matches \\r\\n:")
    test_str = "\r\n"

    components = [
        ("(?i:'s|'t|'re|'ve|'m|'ll|'d)", "contractions"),
        ("[^\r\n\p{L}\p{N}]?\p{L}+", "letters"),
        ("\p{N}", "numbers"),
        (" ?[^\s\p{L}\p{N}]+[\r\n]*", "punctuation"),
        ("\s*[\r\n]+", "newlines"),
        ("\s+", "whitespace"),
    ]

    for pattern, name in components:
        try:
            matches = list(re.compile(pattern).finditer(test_str))
            if matches:
                print(f"  {name}: {[repr(m.group()) for m in matches]}")
        except:
            pass

if __name__ == "__main__":
    test_patterns()