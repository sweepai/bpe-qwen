/// Pre-tokenization module with fast and slow implementations for testing
use regex::Regex;
use fancy_regex::Regex as FancyRegex;
use std::sync::OnceLock;
use ctor::ctor;

/// The Qwen pre-tokenization pattern with lookahead
pub const QWEN_PATTERN_WITH_LOOKAHEAD: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The Qwen pre-tokenization pattern without lookahead
pub const QWEN_PATTERN_WITHOUT_LOOKAHEAD: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+";

/// The Qwen pre-tokenization pattern using \z anchor and horizontal whitespace + letters
pub const QWEN_PATTERN_FAST: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\S\r\n]\p{L}+|[^\s\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+\z|\s+";

/// Globally precompiled regex for maximum performance
pub static GLOBAL_QWEN_REGEX: OnceLock<Regex> = OnceLock::new();

/// Globally precompiled regex for the fast single-pass implementation
pub static GLOBAL_QWEN_FAST_REGEX: OnceLock<Regex> = OnceLock::new();

/// Globally precompiled fancy regex for the slow implementation
static GLOBAL_QWEN_FANCY_REGEX: OnceLock<FancyRegex> = OnceLock::new();

/// Initialize the global regexes automatically on module load
#[ctor]
fn initialize_regexes() {
    // Initialize the OnceLock globals
    GLOBAL_QWEN_REGEX.set(Regex::new(QWEN_PATTERN_WITHOUT_LOOKAHEAD)
        .expect("Failed to compile global Qwen regex")).ok();
    GLOBAL_QWEN_FAST_REGEX.set(Regex::new(QWEN_PATTERN_FAST)
        .expect("Failed to compile global Qwen fast regex")).ok();
    GLOBAL_QWEN_FANCY_REGEX.set(FancyRegex::new(QWEN_PATTERN_WITH_LOOKAHEAD)
        .expect("Failed to compile global Qwen fancy regex")).ok();
}

/// Slow but correct implementation using fancy-regex with lookahead
pub fn pretokenize_slow(text: &str) -> Vec<String> {
    GLOBAL_QWEN_FANCY_REGEX.get()
        .expect("Global regex not initialized")
        .find_iter(text)
        .filter_map(|m| m.ok())
        .map(|m| m.as_str().to_string())
        .collect()
}

/// NEW: merge a standalone trailing `"` into a preceding `"word`
fn merge_double_quotes(tokens: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(tokens.len());
    let mut i = 0;

    while i < tokens.len() {
        if i + 1 < tokens.len() && tokens[i + 1] == "\"" {
            let cur = &tokens[i];

            // allow one leading space before the opening quote (from hspace donation)
            let rest = if let Some(stripped) = cur.strip_prefix(' ') { stripped } else { cur };

            if rest.starts_with('"') {
                // count double quotes inside `rest`
                let quote_count = rest.chars().filter(|&c| c == '"').count();

                // we merge only if there is exactly one `"` so far (i.e., it's an opening quote)
                // and (optionally) previous token does not contain tabs (to mimic your earlier guard)
                let prev_has_tabs = i > 0 && tokens[i - 1].contains('\t');
                if quote_count == 1 && !prev_has_tabs {
                    out.push(format!("{}\"", cur));
                    i += 2;
                    continue;
                }
            }
        }

        out.push(tokens[i].clone());
        i += 1;
    }

    out
}

/// Fast implementation using two-pass approach: regex + contraction fix + horizontal whitespace fusion + quote merging
pub fn pretokenize_fast(text: &str) -> Vec<String> {
    // First pass: use the fast regex to get initial tokens
    let initial = pretokenize_fast_single_pass(text);
    // Debug: for the failing case
    if text.contains("quoted") {
        eprintln!("DEBUG: Text='{}', Initial tokens: {:?}", text, initial);
    }

    // Second pass: fix contractions (opening quotes vs real contractions)
    let fixed = fix_contractions(&initial);
    if text.contains("quoted") {
        eprintln!("DEBUG: After fix_contractions: {:?}", fixed);
    }

    // Third pass: apply horizontal whitespace fusion rules
    let fused = fuse_hspace(&fixed);
    if text.contains("quoted") {
        eprintln!("DEBUG: After fuse_hspace: {:?}", fused);
    }

    // Fourth pass: merge standalone trailing quotes with preceding quoted tokens
    let result = merge_double_quotes(&fused);
    if text.contains("quoted") {
        eprintln!("DEBUG: After merge_double_quotes: {:?}", result);
    }
    result
}

/// Single-pass fast implementation using \z anchor instead of lookahead
pub fn pretokenize_fast_single_pass(text: &str) -> Vec<String> {
    GLOBAL_QWEN_FAST_REGEX.get()
        .expect("Global fast regex not initialized")
        .find_iter(text)
        .map(|m| m.as_str().to_string())
        .collect()
}

/// Helper functions for two-pass approach
fn is_hspace(c: char) -> bool {
    c.is_whitespace() && c != '\n' && c != '\r'
}

fn starts_with_letter(s: &str) -> bool {
    s.chars().next().map(|c| c.is_alphabetic()).unwrap_or(false)
}

fn starts_with_punct(s: &str) -> bool {
    s.chars().next().map(|c| !c.is_whitespace() && !c.is_alphanumeric()).unwrap_or(false)
}

fn is_hspace_only_no_nl(s: &str) -> bool {
    !s.chars().any(|c| c == '\n' || c == '\r') && s.chars().all(is_hspace)
}

fn split_off_last_char(s: &str) -> (&str, &str) {
    if s.is_empty() {
        return ("", "");
    }

    let mut last_char_start = s.len();
    for (i, _) in s.char_indices() {
        last_char_start = i;
    }

    if last_char_start == 0 {
        // Single character
        ("", s)
    } else {
        (&s[..last_char_start], &s[last_char_start..])
    }
}

fn is_alnum_last(s: &str) -> bool {
    s.chars().rev().find(|c| !c.is_whitespace()).map_or(false, |c| c.is_alphanumeric())
}

fn is_contraction(tok: &str) -> bool {
    matches!(&tok.to_ascii_lowercase()[..],
        "'s" | "'t" | "'re" | "'ve" | "'m" | "'ll" | "'d")
}

fn starts_with_letters(s: &str) -> bool {
    s.chars().next().map(|c| c.is_alphabetic()).unwrap_or(false)
}

fn fix_contractions(tokens: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        if i + 1 < tokens.len() && is_contraction(&tokens[i]) && starts_with_letters(&tokens[i + 1]) {
            // treat as contraction only if previous ends with alnum (e.g., I've)
            if !out.last().map_or(false, |p| is_alnum_last(p)) {
                // opening quote case: merge "'ve" + "rbose" => "'verbose"
                out.push(format!("{}{}", tokens[i], tokens[i + 1]));
                i += 2;
                continue;
            }
        }
        out.push(tokens[i].clone());
        i += 1;
    }
    out
}

fn starts_with_digit(s: &str) -> bool {
    s.chars().next().map(|c| c.is_numeric()).unwrap_or(false)
}

fn split_off_last_char_string(s: &str) -> (String, String) {
    if s.is_empty() {
        return (String::new(), String::new());
    }
    let mut last = 0;
    for (i, _) in s.char_indices() {
        last = i;
    }
    if last == 0 {
        (String::new(), s.to_string())
    } else {
        (s[..last].to_string(), s[last..].to_string())
    }
}

fn split_punct_letters(s: &str) -> Option<(&str, &str)> {
    // Our [^\s\p{L}\p{N}]?\p{L}+ ensures at most 1 leading punct before letters.
    let mut it = s.chars();
    let first = it.next()?;
    if !first.is_whitespace() && !first.is_alphanumeric() {
        if it.as_str().chars().next().map(|c| c.is_alphabetic()).unwrap_or(false) {
            return Some((&s[..first.len_utf8()], &s[first.len_utf8()..]));
        }
    }
    None
}

fn fuse_hspace(tokens: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(tokens.len());
    let mut i = 0;

    while i < tokens.len() {
        let cur = &tokens[i];

        let is_ws_no_nl = !cur.contains('\n') && !cur.contains('\r') && cur.chars().all(|c| c.is_whitespace());
        if is_ws_no_nl && i + 1 < tokens.len() {
            let next = &tokens[i + 1];
            let (rest, last) = split_off_last_char_string(cur);
            let last_ch = last.chars().next().unwrap_or('\0');
            let has_rest = !rest.is_empty();

            // CASE A: next starts with letters -> donate one trailing hspace (space or tab)
            if starts_with_letters(next) {
                if has_rest { out.push(rest); }
                out.push(format!("{}{}", last, next));
                i += 2;
                continue;
            }

            // CASE B: next starts with punctuation
            if starts_with_punct(next) {
                // Opening apostrophe followed by letters needs to become " '" + "test"
                if last_ch == ' ' {
                    if let Some((punct, letters)) = split_punct_letters(next) {
                        if punct == "'" {
                            if has_rest { out.push(rest); }
                            out.push(" '".to_string());
                            out.push(letters.to_string());
                            i += 2;
                            continue;
                        }
                    }
                    // Other punctuation: merge the space
                    if has_rest { out.push(rest); }
                    out.push(format!("{}{}", last, next)); // " " + ".filter" => " .filter"
                    i += 2;
                    continue;
                } else if last_ch == '\t' {
                    // Tabs do NOT merge to punctuation: split off as its own token
                    if has_rest { out.push(rest); }
                    out.push("\t".to_string());
                    out.push(next.clone());
                    i += 2;
                    continue;
                }
            }

            // CASE C: next starts with digit -> split off ONE space as its own token
            if starts_with_digit(next) && last_ch == ' ' {
                if has_rest { out.push(rest); }
                out.push(" ".to_string());
                // don't consume next; let loop handle it
                i += 1;
                continue;
            }
        }

        out.push(cur.clone());
        i += 1;
    }
    out
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

        // Check if this match starts with a double quote and we have a closing quote next
        if mat.starts_with('"') && mat.len() > 1 && i + 1 < matches.len() {
            let next = matches[i + 1].as_str();
            if next == "\"" {
                // Check if the previous token was a tab - if so, DON'T merge the quotes
                let should_merge = if i > 0 {
                    let prev = matches[i - 1].as_str();
                    !prev.contains('\t')  // Don't merge if previous token contains tabs
                } else {
                    true  // Merge if no previous token
                };

                if should_merge {
                    // Merge the quoted string with the closing quote
                    result.push(format!("{}{}", mat, next));
                    i += 2;
                    continue;
                }
            }
        }

        // Check if this match is a contraction pattern that might have incorrectly split a word
        if mat == "'s" || mat == "'t" || mat == "'re" || mat == "'ve" ||
            mat == "'m" || mat == "'ll" || mat == "'d" ||
            mat == "'S" || mat == "'T" || mat == "'RE" || mat == "'VE" ||
            mat == "'M" || mat == "'LL" || mat == "'D" {

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
                        // Convert byte position to character position safely
                        let text_up_to_pos = &text[..start_pos];
                        match text_up_to_pos.chars().last() {
                            Some(prev_char) => !prev_char.is_alphanumeric(),
                            None => true, // If we can't get the previous character, treat as non-alphanumeric
                        }
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

            // Special handling for consecutive tabs - split them into individual tokens
            if mat.len() > 1 && mat.chars().all(|c| c == '\t') {
                // Split multiple consecutive tabs into individual tab tokens
                for _ in 0..mat.len() {
                    result.push("\t".to_string());
                }
                i += 1;
                continue;
            }

            // Special handling for mixed whitespace - split each character individually
            if mat.len() > 1 && mat.contains('\t') {
                // Split mixed whitespace (space + tab combinations) into individual tokens
                for c in mat.chars() {
                    result.push(c.to_string());
                }
                i += 1;
                continue;
            }
            // Check what follows this match
            if i + 1 < matches.len() {
                let next = matches[i + 1].as_str();

                // The lookahead \s+(?!\S) means "whitespace NOT followed by non-whitespace"
                // So when whitespace IS followed by non-whitespace, we need to handle it specially
                if !next.is_empty() {
                    let first_char = next.chars().next().unwrap();

                    // Check if this whitespace contains tabs - tabs should NOT be merged with punctuation
                    let contains_tabs = mat.contains('\t');

                    if first_char.is_alphabetic() {
                        // For alphabetic chars, merge the last space with the next token (but not tabs)
                        let space_chars: Vec<char> = mat.chars().collect();

                        if space_chars.len() > 1 && !contains_tabs {
                            // Keep all but last space as separate token (only if no tabs)
                            let first_part: String = space_chars[..space_chars.len() - 1].iter().collect();
                            let last_space: String = space_chars[space_chars.len() - 1].to_string();

                            result.push(first_part);
                            // Merge last space with next token
                            result.push(format!("{}{}", last_space, next));
                            i += 2; // Skip next token since we merged it
                            continue;
                        } else if space_chars.len() == 1 && !contains_tabs {
                            // Single space - merge with next token (only if not a tab)
                            result.push(format!("{}{}", mat, next));
                            i += 2; // Skip next token since we merged it
                            continue;
                        }
                        // If contains tabs, fall through to default case (keep separate)
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
                    } else if !first_char.is_whitespace() && !contains_tabs {
                        // For punctuation/other non-whitespace, non-alphabetic, non-numeric
                        // Only merge with spaces, NOT tabs
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
                    // If we reach here with tabs or other conditions, fall through to default behavior
                }
            }
        }

        // Default case: keep the match as-is
        result.push(mat.to_string());
        i += 1;
    }

    result
}



#[cfg(test)]
mod tests {
    use super::*;

    /// Helper function to compare slow and fast tokenization results
    fn compare_pretokenization(text: &str) -> (Vec<String>, Vec<String>, bool) {
        let slow = pretokenize_slow(text);
        let fast = pretokenize_fast(text);
        let matches = slow == fast;
        (slow, fast, matches)
    }

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