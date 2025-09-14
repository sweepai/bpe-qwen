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

    // Second pass: fix incorrectly split contractions and apply whitespace correction
    let mut result = Vec::with_capacity(matches.len());
    let mut i = 0;

    while i < matches.len() {
        let mat = matches[i].as_str();

        // Check if this match is a contraction pattern that might have incorrectly split a word
        if (mat == "'s" || mat == "'t" || mat == "'re" || mat == "'ve" ||
            mat == "'m" || mat == "'ll" || mat == "'d" ||
            mat == "'S" || mat == "'T" || mat == "'RE" || mat == "'VE" ||
            mat == "'M" || mat == "'LL" || mat == "'D") {

            // Check if the next token starts with letters (indicating it was incorrectly split)
            if i + 1 < matches.len() {
                let next = matches[i + 1].as_str();

                // If next token starts with a letter, this was likely an incorrect split
                if !next.is_empty() && next.chars().next().unwrap().is_alphabetic() {
                    // Check what came before this token to determine if it's a real contraction
                    // or an incorrectly split word like 'verbose'
                    let start_pos = matches[i].start();

                    // If there's nothing before the apostrophe, or it's whitespace/punctuation,
                    // this is a quoted word, not a contraction
                    if start_pos == 0 || (start_pos > 0 && {
                        let prev_char = text.chars().nth(start_pos - 1).unwrap();
                        !prev_char.is_alphanumeric()
                    }) {
                        // Check if we just added a whitespace token that should have the quote attached
                        if !result.is_empty() {
                            let last_token = result.last().unwrap();
                            if last_token == " " {
                                // Pop the single space and replace with space + quote
                                result.pop();
                                result.push(" '".to_string());
                                // Add the full word (contraction pattern + letters)
                                result.push(format!("{}{}", mat.chars().skip(1).collect::<String>(), next));
                                i += 2;
                                continue;
                            }
                        }

                        // Otherwise just merge the incorrectly split tokens
                        result.push(format!("{}{}", mat, next));
                        i += 2;
                        continue;
                    }
                }
            }
        }

        // Check if this match is only whitespace (but not containing \r or \n)
        // \r and \n are handled specially by the regex and should not be split
        if mat.chars().all(|c| c.is_whitespace())
            && !mat.contains('\r')
            && !mat.contains('\n') {
            // Check what follows this match
            if i + 1 < matches.len() {
                let next = matches[i + 1].as_str();

                // The lookahead \s+(?!\S) means "whitespace NOT followed by non-whitespace"
                // So when whitespace IS followed by non-whitespace, we need to handle it specially
                if !next.is_empty() {
                    let first_char = next.chars().next().unwrap();

                    if first_char.is_alphabetic() {
                        // For alphabetic chars, merge the last space with the next token
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
                    } else if first_char.is_numeric() {
                        // For numbers, split whitespace but keep them separate
                        let space_chars: Vec<char> = mat.chars().collect();

                        if space_chars.len() > 1 {
                            // Split into n-1 spaces and 1 space (like slow version)
                            let first_part: String = space_chars[..space_chars.len() - 1].iter().collect();
                            let last_space: String = space_chars[space_chars.len() - 1].to_string();

                            result.push(first_part);
                            result.push(last_space);
                            i += 1; // Continue to process next token normally
                            continue;
                        }
                    } else if !first_char.is_whitespace() {
                        // For punctuation/other non-whitespace, non-alphabetic, non-numeric
                        // The pattern ` ?[^\s\p{L}\p{N}]+` matches space + punctuation
                        // But we need to be careful not to create tokens that would be split by contractions

                        // Check if next token starts with a single quote and could be a contraction
                        if next.starts_with('\'') && next.len() > 1 {
                            // Check if this could match a contraction pattern
                            let next_lower = next.to_lowercase();
                            if next_lower.starts_with("'s") || next_lower.starts_with("'t") ||
                               next_lower.starts_with("'re") || next_lower.starts_with("'ve") ||
                               next_lower.starts_with("'m") || next_lower.starts_with("'ll") ||
                               next_lower.starts_with("'d") {
                                // This is likely a contraction, don't merge space with it
                                // Just split the whitespace normally
                                let space_chars: Vec<char> = mat.chars().collect();
                                if space_chars.len() > 1 {
                                    let first_part: String = space_chars[..space_chars.len() - 1].iter().collect();
                                    let last_space: String = space_chars[space_chars.len() - 1].to_string();
                                    result.push(first_part);
                                    result.push(last_space);
                                    i += 1;
                                    continue;
                                }
                            } else {
                                // Check if it's punctuation followed by letters that could form 've', 're', etc.
                                let has_letters = next.chars().skip(1).any(|c| c.is_alphabetic());

                                if has_letters {
                                    // This could be like 'verbose' - check if it would match contractions
                                    let letter_part: String = next.chars().skip(1).collect();
                                    if letter_part.to_lowercase().starts_with("ve") ||
                                       letter_part.to_lowercase().starts_with("re") ||
                                       letter_part.to_lowercase().starts_with("ll") ||
                                       letter_part.to_lowercase().starts_with("s") ||
                                       letter_part.to_lowercase().starts_with("t") ||
                                       letter_part.to_lowercase().starts_with("m") ||
                                       letter_part.to_lowercase().starts_with("d") {
                                        // Would match a contraction pattern - keep them separate
                                        let space_chars: Vec<char> = mat.chars().collect();
                                        if space_chars.len() > 1 {
                                            let first_part: String = space_chars[..space_chars.len() - 1].iter().collect();
                                            let last_space: String = space_chars[space_chars.len() - 1].to_string();
                                            result.push(first_part);
                                            // Merge space with just the punctuation, not the letters
                                            result.push(format!("{}{}", last_space, "'"));
                                            // Add the letters separately
                                            result.push(letter_part);
                                            i += 2;
                                            continue;
                                        } else {
                                            // Single space - merge with punctuation only
                                            result.push(format!("{}{}", mat, "'"));
                                            result.push(letter_part);
                                            i += 2;
                                            continue;
                                        }
                                    }
                                }
                            }
                        }

                        // For other punctuation cases
                        let next_chars: Vec<char> = next.chars().collect();
                        let has_letters = next_chars.iter().skip(1).any(|c| c.is_alphabetic());

                        if !first_char.is_alphabetic() && has_letters {
                            // Punctuation followed by letters (but not contractions)
                            let space_chars: Vec<char> = mat.chars().collect();

                            if space_chars.len() > 1 {
                                let first_part: String = space_chars[..space_chars.len() - 1].iter().collect();
                                let last_space: String = space_chars[space_chars.len() - 1].to_string();

                                result.push(first_part);

                                // Split punctuation from letters
                                let letter_start = next_chars.iter().position(|c| c.is_alphabetic()).unwrap_or(next_chars.len());
                                if letter_start > 0 && letter_start < next_chars.len() {
                                    let punct_part: String = next_chars[..letter_start].iter().collect();
                                    let letter_part: String = next_chars[letter_start..].iter().collect();
                                    result.push(format!("{}{}", last_space, punct_part));
                                    result.push(letter_part);
                                } else {
                                    result.push(format!("{}{}", last_space, next));
                                }
                                i += 2;
                                continue;
                            } else {
                                // Single space
                                let letter_start = next_chars.iter().position(|c| c.is_alphabetic()).unwrap_or(next_chars.len());
                                if letter_start > 0 && letter_start < next_chars.len() {
                                    let punct_part: String = next_chars[..letter_start].iter().collect();
                                    let letter_part: String = next_chars[letter_start..].iter().collect();
                                    result.push(format!("{}{}", mat, punct_part));
                                    result.push(letter_part);
                                } else {
                                    result.push(format!("{}{}", mat, next));
                                }
                                i += 2;
                                continue;
                            }
                        } else if !next.starts_with(' ') {
                            // Pure punctuation without letters
                            let space_chars: Vec<char> = mat.chars().collect();

                            if space_chars.len() > 1 {
                                let first_part: String = space_chars[..space_chars.len() - 1].iter().collect();
                                let last_space: String = space_chars[space_chars.len() - 1].to_string();

                                result.push(first_part);
                                result.push(format!("{}{}", last_space, next));
                                i += 2;
                                continue;
                            } else {
                                result.push(format!("{}{}", mat, next));
                                i += 2;
                                continue;
                            }
                        }
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