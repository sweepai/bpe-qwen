/// Pre-tokenization module with fast and slow implementations for testing
use regex::Regex;
use fancy_regex::Regex as FancyRegex;

/// The Qwen pre-tokenization pattern with lookahead
pub const QWEN_PATTERN_WITH_LOOKAHEAD: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The Qwen pre-tokenization pattern without lookahead
pub const QWEN_PATTERN_WITHOUT_LOOKAHEAD: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+";

/// Slow but correct implementation using fancy-regex with lookahead
pub fn pretokenize_slow(text: &str) -> Vec<String> {
    let regex = FancyRegex::new(QWEN_PATTERN_WITH_LOOKAHEAD)
        .expect("Failed to compile fancy regex");

    regex.find_iter(text)
        .filter_map(|m| m.ok())
        .map(|m| m.as_str().to_string())
        .collect()
}

/// Fast implementation using standard regex with correction pass
pub fn pretokenize_fast(text: &str) -> Vec<String> {
    let regex = Regex::new(QWEN_PATTERN_WITHOUT_LOOKAHEAD)
        .expect("Failed to compile standard regex");
    pretokenize_fast_with_regex(text, &regex)
}

/// Fast implementation using a pre-compiled regex (for performance)
pub fn pretokenize_fast_with_regex(text: &str, regex: &Regex) -> Vec<String> {
    // First pass: collect all regex matches
    let matches: Vec<_> = regex.find_iter(text).collect();

    // Second pass: apply whitespace correction to mimic lookahead behavior
    // The pattern \s+(?!\S) means "whitespace not followed by non-whitespace"
    // When whitespace is followed by non-whitespace, the last space attaches to the next token
    let mut result = Vec::with_capacity(matches.len());
    let mut i = 0;

    while i < matches.len() {
        let mat = matches[i].as_str();

        // Check if this match is only whitespace (but not containing \r or \n)
        // \r and \n are handled specially by the regex and should not be split
        if mat.chars().all(|c| c.is_whitespace())
            && !mat.contains('\r')
            && !mat.contains('\n') {
            // Check what follows this match
            if i + 1 < matches.len() {
                let next = matches[i + 1].as_str();

                // If next token starts with non-whitespace, merge last space with it
                if !next.is_empty() && !next.chars().next().unwrap().is_whitespace() {
                    let space_chars: Vec<char> = mat.chars().collect();

                    if space_chars.len() > 1 {
                        // Keep all but last space as separate token
                        let first_part: String = space_chars[..space_chars.len() - 1].iter().collect();
                        let last_space: String = space_chars[space_chars.len() - 1].to_string();

                        result.push(first_part);
                        // Merge last space with next token
                        result.push(format!("{}{}", last_space, next));
                        i += 2; // Skip next token since we merged it
                        continue;
                    } else {
                        // Single space - merge with next token
                        result.push(format!("{}{}", mat, next));
                        i += 2; // Skip next token since we merged it
                        continue;
                    }
                }
            }
        }

        // Default case: keep the match as-is
        result.push(mat.to_string());
        i += 1;
    }

    result
}

/// Test helper to compare outputs of fast and slow implementations
pub fn compare_pretokenization(text: &str) -> (Vec<String>, Vec<String>, bool) {
    let slow_result = pretokenize_slow(text);
    let fast_result = pretokenize_fast(text);
    let matches = slow_result == fast_result;
    (slow_result, fast_result, matches)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_text() {
        let text = "Hello, world!";
        let (slow, fast, matches) = compare_pretokenization(text);
        assert!(matches, "Mismatch for '{}'\nSlow: {:?}\nFast: {:?}", text, slow, fast);
    }

    #[test]
    fn test_leading_spaces() {
        let text = "   leading spaces";
        let (slow, fast, matches) = compare_pretokenization(text);
        assert!(matches, "Mismatch for '{}'\nSlow: {:?}\nFast: {:?}", text, slow, fast);
    }

    #[test]
    fn test_multiple_spaces() {
        let text = "word1  word2   word3";
        let (slow, fast, matches) = compare_pretokenization(text);
        assert!(matches, "Mismatch for '{}'\nSlow: {:?}\nFast: {:?}", text, slow, fast);
    }

    #[test]
    fn test_trailing_spaces() {
        let text = "word   ";
        let (slow, fast, matches) = compare_pretokenization(text);
        assert!(matches, "Mismatch for '{}'\nSlow: {:?}\nFast: {:?}", text, slow, fast);
    }

    #[test]
    fn test_code_indentation() {
        let text = "def hello():\n    print('hi')";
        let (slow, fast, matches) = compare_pretokenization(text);
        assert!(matches, "Mismatch for '{}'\nSlow: {:?}\nFast: {:?}", text, slow, fast);
    }

    #[test]
    fn test_mixed_whitespace() {
        let text = "  \n  text";
        let (slow, fast, matches) = compare_pretokenization(text);
        assert!(matches, "Mismatch for '{}'\nSlow: {:?}\nFast: {:?}", text, slow, fast);
    }

    #[test]
    fn test_single_space() {
        let text = " a";
        let (slow, fast, matches) = compare_pretokenization(text);
        assert!(matches, "Mismatch for '{}'\nSlow: {:?}\nFast: {:?}", text, slow, fast);
    }

    #[test]
    fn test_double_space() {
        let text = "  a";
        let (slow, fast, matches) = compare_pretokenization(text);
        assert!(matches, "Mismatch for '{}'\nSlow: {:?}\nFast: {:?}", text, slow, fast);
    }

    #[test]
    fn test_tabs() {
        let text = "\t\tcode";
        let (slow, fast, matches) = compare_pretokenization(text);
        assert!(matches, "Mismatch for '{}'\nSlow: {:?}\nFast: {:?}", text, slow, fast);
    }

    #[test]
    fn test_crlf() {
        let text = "line1\r\nline2";
        let (slow, fast, matches) = compare_pretokenization(text);
        assert!(matches, "Mismatch for '{}'\nSlow: {:?}\nFast: {:?}", text, slow, fast);
    }

    #[test]
    fn test_mixed_line_endings() {
        let text = "line1\rline2\nline3\r\nline4";
        let (slow, fast, matches) = compare_pretokenization(text);

        // Print detailed output for debugging
        println!("Text: {:?}", text);
        println!("Slow result: {:?}", slow);
        println!("Fast result: {:?}", fast);
        println!("Matches: {}", matches);

        // Check specifically for the \r\n sequence
        let has_crlf_slow = slow.iter().any(|s| s == "\r\n");
        let has_crlf_fast = fast.iter().any(|s| s == "\r\n");
        println!("Slow has \\r\\n token: {}", has_crlf_slow);
        println!("Fast has \\r\\n token: {}", has_crlf_fast);

        assert!(matches, "Mismatch for '{}'\nSlow: {:?}\nFast: {:?}", text, slow, fast);
    }
}