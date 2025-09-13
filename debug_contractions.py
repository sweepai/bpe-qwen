import regex

# Test the regex patterns
pattern_with_lookahead = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
pattern_without_lookahead = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+"

test_text = "'verbose'"

print("Test text:", repr(test_text))
print()

regex_with = regex.compile(pattern_with_lookahead)
regex_without = regex.compile(pattern_without_lookahead)

print("With lookahead:")
matches_with = regex_with.findall(test_text)
for i, match in enumerate(matches_with):
    print(f"  {i}: {repr(match)}")

print("\nWithout lookahead:")
matches_without = regex_without.findall(test_text)
for i, match in enumerate(matches_without):
    print(f"  {i}: {repr(match)}")

# Test another case
test_text2 = "I've got"
print(f"\n\nTest text 2: {repr(test_text2)}")

print("\nWith lookahead:")
matches_with2 = regex_with.findall(test_text2)
for i, match in enumerate(matches_with2):
    print(f"  {i}: {repr(match)}")

print("\nWithout lookahead:")
matches_without2 = regex_without.findall(test_text2)
for i, match in enumerate(matches_without2):
    print(f"  {i}: {repr(match)}")