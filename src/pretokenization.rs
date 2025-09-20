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

    // Second pass: fix contractions (opening quotes vs real contractions)
    let fixed = fix_contractions(&initial);

    // Third pass: apply horizontal whitespace fusion rules
    let fused = fuse_hspace(&fixed);

    // Fourth pass: merge standalone trailing quotes with preceding quoted tokens
    merge_double_quotes(&fused)
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

fn starts_with_punct(s: &str) -> bool {
    s.chars().next().map(|c| !c.is_whitespace() && !c.is_alphanumeric()).unwrap_or(false)
}

fn is_hspace_only_no_nl(s: &str) -> bool {
    !s.chars().any(|c| c == '\n' || c == '\r') && s.chars().all(is_hspace)
}

fn split_off_last_char(s: &str) -> (String, String) {
    if s.is_empty() { return (String::new(), String::new()); }
    let mut last = 0;
    for (i, _) in s.char_indices() { last = i; }
    if last == 0 { (String::new(), s.to_string()) } else { (s[..last].to_string(), s[last..].to_string()) }
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

        let is_ws_no_nl = is_hspace_only_no_nl(cur);
        if is_ws_no_nl && i + 1 < tokens.len() {
            let next = &tokens[i + 1];
            let (rest, last) = split_off_last_char(cur);
            let last_ch = last.chars().next().unwrap_or('\0');
            let has_rest = !rest.is_empty();

            // CASE A: next starts with letters -> donate one trailing hspace (space or tab)
            if starts_with_letters(next) {
                if has_rest { out.push(rest); }
                out.push(format!("{}{}", last, next)); // " world" / "\tcode"
                i += 2;
                continue;
            }

            // CASE B: next starts with punctuation
            if starts_with_punct(next) {
                if last_ch == ' ' {
                    // Apostrophe opener: " '" + "word"
                    if let Some((punct, letters)) = split_punct_letters(next) {
                        if punct == "'" {
                            if has_rest { out.push(rest); }
                            out.push(" '".to_string());
                            out.push(letters.to_string());
                            i += 2;
                            continue;
                        } else {
                            // General punct+letters: " <" + "Select", " ." + "filter", " @" + "NotNull", " \"" + "target"
                            if has_rest { out.push(rest); }
                            out.push(format!(" {}", punct));
                            out.push(letters.to_string());
                            i += 2;
                            continue;
                        }
                    }
                    // Pure punctuation (no letters): keep the merge (e.g. " */")
                    if has_rest { out.push(rest); }
                    out.push(format!("{}{}", last, next));
                    i += 2;
                    continue;
                } else if last_ch == '\t' {
                    // Tabs never merge into punctuation
                    if has_rest { out.push(rest); }
                    out.push("\t".to_string());
                    out.push(next.clone());
                    i += 2;
                    continue;
                }
            }

            // CASE C: digits — never glue space to numbers; split off one space
            if starts_with_digit(next) && last_ch == ' ' {
                if has_rest { out.push(rest); }
                out.push(" ".to_string());
                i += 1; // let loop handle the number token next
                continue;
            }
        }

        out.push(cur.clone());
        i += 1;
    }
    out
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

    #[test]
    fn test_sft_space_punct_letters() {
        // Test cases for SFT tokenization fix: space + punctuation + letters
        assert_eq!(pretokenize_fast(" <Select"), vec![" <", "Select"]);
        assert_eq!(pretokenize_fast(" (options"), vec![" (", "options"]);
        assert_eq!(pretokenize_fast(" @NotNull"), vec![" @", "NotNull"]);
        assert_eq!(pretokenize_fast(" .filter"), vec![" .", "filter"]);
        assert_eq!(pretokenize_fast(" \"target"), vec![" \"", "target"]);
    }
}