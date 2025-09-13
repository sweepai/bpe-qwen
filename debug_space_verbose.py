import regex

# Test the regex patterns
pattern_with_lookahead = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
pattern_without_lookahead = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+"

test_text = " 'verbose'"

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

# Now let's understand the pattern better
print("\n\nPattern analysis:")
print("The pattern ` ?[^\\s\\p{L}\\p{N}]+` matches punctuation with optional leading space")
print("So ` '` would be matched by this pattern")
print("\nThe pattern `[^\\r\\n\\p{L}\\p{N}]?\\p{L}+` matches optional punctuation + letters")
print("So `'verbose` would be matched by this pattern")
print("\nBut the contraction pattern `(?i:'s|'t|'re|'ve|'m|'ll|'d)` has higher priority")
print("So it matches `'ve` first, leaving `rbose` for the next match")