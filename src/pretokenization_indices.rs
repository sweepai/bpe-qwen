use crate::pretokenization::GLOBAL_QWEN_FAST_REGEX;

/// Single-pass fast implementation using \z anchor instead of lookahead (indices version)
fn pretokenize_fast_single_pass_indices(text: &str) -> Vec<usize> {
    GLOBAL_QWEN_FAST_REGEX.get()
        .expect("Global fast regex not initialized")
        .find_iter(text)
        .map(|m| m.end())
        .collect()
}

/// Helper functions for indices-based operations
fn get_token_at_index<'a>(text: &'a str, end_indices: &[usize], index: usize) -> &'a str {
    let start = if index == 0 { 0 } else { end_indices[index - 1] };
    let end = end_indices[index];
    &text[start..end]
}

fn is_alnum_last_indices(text: &str, end_indices: &[usize], index: usize) -> bool {
    let start = if index == 0 { 0 } else { end_indices[index - 1] };
    let end = end_indices[index];
    if start >= end { return false; }

    // Get last non-whitespace character directly
    text[start..end].chars().rev().find(|c| !c.is_whitespace()).map_or(false, |c| c.is_alphanumeric())
}

fn is_contraction_indices(text: &str, end_indices: &[usize], index: usize) -> bool {
    let start = if index == 0 { 0 } else { end_indices[index - 1] };
    let end = end_indices[index];
    let token_bytes = &text.as_bytes()[start..end];

    // Fast direct byte comparison without string allocation
    match token_bytes {
        [b'\'', b's'] | [b'\'', b'S'] => true,
        [b'\'', b't'] | [b'\'', b'T'] => true,
        [b'\'', b'm'] | [b'\'', b'M'] => true,
        [b'\'', b'd'] | [b'\'', b'D'] => true,
        [b'\'', b'r', b'e'] | [b'\'', b'R', b'e'] | [b'\'', b'r', b'E'] | [b'\'', b'R', b'E'] => true,
        [b'\'', b'v', b'e'] | [b'\'', b'V', b'e'] | [b'\'', b'v', b'E'] | [b'\'', b'V', b'E'] => true,
        [b'\'', b'l', b'l'] | [b'\'', b'L', b'l'] | [b'\'', b'l', b'L'] | [b'\'', b'L', b'L'] => true,
        _ => false,
    }
}

fn starts_with_letters_indices(text: &str, end_indices: &[usize], index: usize) -> bool {
    let start = if index == 0 { 0 } else { end_indices[index - 1] };
    let end = end_indices[index];
    if start >= end { return false; }

    // Get first character directly without string slice
    text[start..].chars().next().map(|c| c.is_alphabetic()).unwrap_or(false)
}

/// Fix contractions (opening quotes vs real contractions) - indices version
fn fix_contractions_indices(text: &str, end_indices: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(end_indices.len());
    let mut i = 0;

    while i < end_indices.len() {
        if i + 1 < end_indices.len()
            && is_contraction_indices(text, end_indices, i)
            && starts_with_letters_indices(text, end_indices, i + 1) {

            // treat as contraction only if previous ends with alnum (e.g., I've)
            if !out.is_empty() && is_alnum_last_indices(text, &out, out.len() - 1) {
                // Real contraction - keep separate
                out.push(end_indices[i]);
                i += 1;
                continue;
            } else {
                // opening quote case: merge "'ve" + "rbose" => "'verbose"
                out.push(end_indices[i + 1]);
                i += 2;
                continue;
            }
        }
        out.push(end_indices[i]);
        i += 1;
    }
    out
}

/// Apply horizontal whitespace fusion rules - indices version
fn fuse_hspace_indices(text: &str, end_indices: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(end_indices.len());
    let mut i = 0;

    while i < end_indices.len() {
        let cur_token = get_token_at_index(text, end_indices, i);

        let is_ws_no_nl = !cur_token.contains('\n') && !cur_token.contains('\r') && cur_token.chars().all(|c| c.is_whitespace());
        if is_ws_no_nl && i + 1 < end_indices.len() {
            let next_token = get_token_at_index(text, end_indices, i + 1);
            let last_ch = cur_token.chars().last().unwrap_or('\0');
            let has_rest = cur_token.len() > 1;

            // CASE A: next starts with letters -> donate one trailing hspace (space or tab)
            if starts_with_letters_indices(text, end_indices, i + 1) {
                if has_rest {
                    // Push all but last char
                    let split_pos = end_indices[i] - last_ch.len_utf8();
                    out.push(split_pos);
                }
                // Merge last char with next token
                out.push(end_indices[i + 1]);
                i += 2;
                continue;
            }

            // CASE B: next starts with punctuation
            if next_token.chars().next().map(|c| !c.is_whitespace() && !c.is_alphanumeric()).unwrap_or(false) {
                // Opening apostrophe followed by letters needs to become " '" + "test"
                if last_ch == ' ' {
                    // Check if it's punct + letters pattern like 'test
                    let mut next_chars = next_token.chars();
                    let first_char = next_chars.next().unwrap_or('\0');
                    if first_char == '\'' && next_chars.as_str().chars().next().map(|c| c.is_alphabetic()).unwrap_or(false) {
                        if has_rest {
                            let split_pos = end_indices[i] - last_ch.len_utf8();
                            out.push(split_pos);
                        }
                        // Push " '"
                        let space_quote_end = end_indices[i] - last_ch.len_utf8() + " '".len();
                        out.push(space_quote_end);
                        // Push the letters part
                        out.push(end_indices[i + 1]);
                        i += 2;
                        continue;
                    }
                    // Other punctuation: merge the space
                    if has_rest {
                        let split_pos = end_indices[i] - last_ch.len_utf8();
                        out.push(split_pos);
                    }
                    out.push(end_indices[i + 1]); // merge space + punctuation
                    i += 2;
                    continue;
                } else if last_ch == '\t' {
                    // Tabs do NOT merge to punctuation: split off as its own token
                    if has_rest {
                        let split_pos = end_indices[i] - last_ch.len_utf8();
                        out.push(split_pos);
                    }
                    out.push(end_indices[i]); // tab alone
                    out.push(end_indices[i + 1]); // punctuation alone
                    i += 2;
                    continue;
                }
            }

            // CASE C: next starts with digit -> split off ONE space as its own token
            if next_token.chars().next().map(|c| c.is_numeric()).unwrap_or(false) && last_ch == ' ' {
                if has_rest {
                    let split_pos = end_indices[i] - last_ch.len_utf8();
                    out.push(split_pos);
                }
                out.push(end_indices[i]); // single space
                // don't consume next; let loop handle it
                i += 1;
                continue;
            }
        }

        out.push(end_indices[i]);
        i += 1;
    }
    out
}

/// Merge standalone trailing quotes with preceding quoted tokens - indices version
fn merge_double_quotes_indices(text: &str, end_indices: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(end_indices.len());
    let mut i = 0;

    while i < end_indices.len() {
        if i + 1 < end_indices.len() {
            let next_token = get_token_at_index(text, end_indices, i + 1);
            if next_token == "\"" {
                let cur_token = get_token_at_index(text, end_indices, i);

                // allow one leading space before the opening quote (from hspace donation)
                let rest = if let Some(stripped) = cur_token.strip_prefix(' ') { stripped } else { cur_token };

                if rest.starts_with('"') {
                    // count double quotes inside `rest`
                    let quote_count = rest.chars().filter(|&c| c == '"').count();

                    // we merge only if there is exactly one `"` so far (i.e., it's an opening quote)
                    // and (optionally) previous token does not contain tabs (to mimic your earlier guard)
                    let prev_has_tabs = if i > 0 {
                        let prev_token = get_token_at_index(text, &out, out.len() - 1);
                        prev_token.contains('\t')
                    } else {
                        false
                    };

                    if quote_count == 1 && !prev_has_tabs {
                        // Merge: extend current token to include the closing quote
                        out.push(end_indices[i + 1]);
                        i += 2;
                        continue;
                    }
                }
            }
        }

        out.push(end_indices[i]);
        i += 1;
    }

    out
}

/// Fast pretokenization that returns only end positions of tokens
/// Returns only end positions since start of next token = end of previous token
/// Uses native four-pass approach entirely in indices space for maximum efficiency
pub fn pretokenize_fast_indices(text: &str) -> Vec<usize> {
    // First pass: use the fast regex to get initial token end indices
    let initial = pretokenize_fast_single_pass_indices(text);

    // Second pass: fix contractions (opening quotes vs real contractions)
    let fixed = fix_contractions_indices(text, &initial);

    // Third pass: apply horizontal whitespace fusion rules
    let fused = fuse_hspace_indices(text, &fixed);

    // Fourth pass: merge standalone trailing quotes with preceding quoted tokens
    merge_double_quotes_indices(text, &fused)
}



/// Convert end indices to actual string slices for testing
pub fn indices_to_strings(text: &str, end_indices: &[usize]) -> Vec<String> {
    if end_indices.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(end_indices.len());
    let mut start = 0;

    for &end in end_indices {
        result.push(text[start..end].to_string());
        start = end;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indices_basic() {
        let text = "Hello world";
        let indices = pretokenize_fast_indices(text);
        let strings = indices_to_strings(text, &indices);
        assert_eq!(strings, vec!["Hello", " world"]);
    }

    #[test]
    fn test_indices_numbers() {
        let text = "    1234";
        let indices = pretokenize_fast_indices(text);
        let strings = indices_to_strings(text, &indices);
        assert_eq!(strings, vec!["   ", " ", "1", "2", "3", "4"]);
    }

    #[test]
    fn test_indices_contractions() {
        let text = "I've got";
        let indices = pretokenize_fast_indices(text);
        let strings = indices_to_strings(text, &indices);
        assert_eq!(strings, vec!["I", "'ve", " got"]);
    }
}