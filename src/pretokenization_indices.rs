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

fn token_is_ascii_spaces_only(text: &str, end_indices: &[usize], index: usize) -> bool {
    if index >= end_indices.len() { return false; }
    let start = if index == 0 { 0 } else { end_indices[index - 1] };
    let end = end_indices[index];
    if start >= end { return false; }
    text[start..end].chars().all(|c| c == ' ')
}

fn starts_with_letters_indices(text: &str, end_indices: &[usize], index: usize) -> bool {
    let start = if index == 0 { 0 } else { end_indices[index - 1] };
    let end = end_indices[index];
    if start >= end { return false; }

    // Use bounded slice for robustness
    text[start..end].chars().next().map(|c| c.is_alphabetic()).unwrap_or(false)
}

fn starts_with_ascii_digit(s: &str) -> bool {
    s.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false)
}

fn starts_with_non_ascii_numeric(s: &str) -> bool {
    s.chars().next().map(|c| c.is_numeric() && !c.is_ascii_digit()).unwrap_or(false)
}

/// Split a token that starts with punctuation followed by letters
/// Returns (punctuation_part, letters_part) if the pattern matches, None otherwise
fn split_punct_letters(token: &str) -> Option<(&str, &str)> {
    let mut chars = token.chars();
    let first_char = chars.next()?;

    // Check if first character is punctuation (not whitespace, not alphanumeric)
    if first_char.is_whitespace() || first_char.is_alphanumeric() {
        return None;
    }

    let rest = chars.as_str();

    // Check if the rest starts with letters
    if rest.chars().next().map(|c| c.is_alphabetic()).unwrap_or(false) {
        let punct_len = first_char.len_utf8();
        Some((&token[..punct_len], &token[punct_len..]))
    } else {
        None
    }
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
                // Opening quote case: merge only in safe contexts.
                // To better match slow tokenizer in code/tab contexts, avoid merging
                // when the preceding token contains tabs/newlines or is non-space whitespace.
                let allow_merge_opening = if out.is_empty() {
                    true // start-of-string
                } else {
                    let prev_is_spaces_only = token_is_ascii_spaces_only(text, &out, out.len() - 1);
                    let prev_tok = get_token_at_index(text, &out, out.len() - 1);
                    let prev_has_tab_or_nl = prev_tok.contains('\t') || prev_tok.contains('\n') || prev_tok.contains('\r');
                    // Allow only if previous is ASCII spaces only (no tabs/newlines)
                    prev_is_spaces_only && !prev_has_tab_or_nl
                };

                if allow_merge_opening {
                    // merge "'ve" + "rbose" => "'verbose"
                    out.push(end_indices[i + 1]);
                    i += 2;
                    continue;
                } else {
                    // do not merge; keep as split to mimic slow behavior in these contexts
                    out.push(end_indices[i]);
                    i += 1;
                    continue;
                }
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
            // Important: determine if there is more than one character, not bytes.
            let has_rest = cur_token.chars().count() > 1;

            // Helper variables for next token analysis
            let next_first = next_token.chars().next();
            let next_starts_letter = starts_with_letters_indices(text, end_indices, i + 1);
            let next_starts_non_ascii_numeric = starts_with_non_ascii_numeric(next_token);

            // CASE A: letters OR non-ASCII numerics (e.g., Ⅻ) -> donate one trailing hspace
            if next_starts_letter || next_starts_non_ascii_numeric {
                if has_rest {
                    let split_pos = end_indices[i] - last_ch.len_utf8();
                    out.push(split_pos);
                }
                out.push(end_indices[i + 1]);
                i += 2;
                continue;
            }

            // CASE B: next starts with punctuation
            if next_token.chars().next().map(|c| !c.is_whitespace() && !c.is_alphanumeric()).unwrap_or(false) {
                if last_ch == ' ' {
                    // If next is "punct + letters" (e.g. ".filter", "<Select", "@NotNull", "\"target")
                    if let Some((punct, _letters)) = split_punct_letters(next_token) {
                        if punct == "'" {
                            // Opening apostrophe + letters => " '" , "word"
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
                        } else {
                            // General case: split punctuation from letters and attach the single space to the punct
                            if has_rest {
                                let split_pos = end_indices[i] - last_ch.len_utf8();
                                out.push(split_pos);
                            }
                            // Push " " + punct (e.g., " .", " <", " @", " \"")
                            let space_punct_end = end_indices[i] - last_ch.len_utf8() + " ".len() + punct.len();
                            out.push(space_punct_end);
                            // Push the letters part
                            out.push(end_indices[i + 1]);
                            i += 2;
                            continue;
                        }
                    }

                    // Otherwise: pure punctuation (no letters) — keep old behavior, merge the space
                    if has_rest {
                        let split_pos = end_indices[i] - last_ch.len_utf8();
                        out.push(split_pos);
                    }
                    out.push(end_indices[i + 1]); // merge space + punctuation
                    i += 2;
                    continue;
                } else if last_ch == '\t' {
                    // Tabs do NOT merge to punctuation: keep a standalone "\t"
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

            // CASE C: next starts with ASCII digit
            if next_first.map(|c| c.is_ascii_digit()).unwrap_or(false) {
                if last_ch == ' ' {
                    if has_rest {
                        let split_pos = end_indices[i] - last_ch.len_utf8();
                        out.push(split_pos);
                    }
                    out.push(end_indices[i]); // single space
                    i += 1;                   // number handled next
                    continue;
                }
                // Tabs before digits: split into (tabs_except_last) + (single '\t'), then digit
                if last_ch == '\t' && cur_token.chars().all(|c| c == '\t') {
                    if has_rest {
                        let split_pos = end_indices[i] - last_ch.len_utf8();
                        out.push(split_pos);      // "\t\t... (n-1)"
                    }
                    out.push(end_indices[i]);     // last "\t"
                    i += 1;                       // digit handled next
                    continue;
                }
            }
        }

        out.push(end_indices[i]);
        i += 1;
    }
    out
}

/// Merge standalone trailing quotes with preceding quoted tokens - generic version
fn merge_trailing_quote_indices(text: &str, end_indices: &[usize], quote: char) -> Vec<usize> {
    let mut out = Vec::with_capacity(end_indices.len());
    let mut i = 0;

    while i < end_indices.len() {
        if i + 1 < end_indices.len() {
            let next = get_token_at_index(text, end_indices, i + 1);
            if next.len() == quote.len_utf8() && next.chars().next() == Some(quote) {
                let cur = get_token_at_index(text, end_indices, i);

                // Only pair if previous token is whitespace or we're at start
                let prev_is_ws = i == 0 || out.last().map_or(false, |&_prev_end| {
                    let prev_tok = get_token_at_index(text, &out, out.len() - 1);
                    prev_tok.chars().all(|c| c.is_whitespace())
                });

                // Guard: do not merge a trailing quote if the preceding token contains tabs.
                // This mirrors the slow/fancy behavior observed in real code and reduces
                // mismatches in tab-indented contexts like: \t"preview" / \t"wayland".
                let prev_has_tabs = if i == 0 { false } else { out.last().map_or(false, |&_prev_end| {
                    let prev_tok = get_token_at_index(text, &out, out.len() - 1);
                    prev_tok.contains('\t')
                }) };

                // Allow a leading donated space
                let cur_stripped = cur.strip_prefix(' ').unwrap_or(cur);

                if prev_is_ws && cur_stripped.starts_with(quote) {
                    if prev_has_tabs {
                        // Keep as-is if previous token had tabs
                        out.push(end_indices[i]);
                        i += 1;
                        continue;
                    }

                    // --- NEW GUARD 1: avoid merging bare contractions (you already had this) ---
                    let is_bare_contr = matches!(cur_stripped.as_bytes(),
                        b"\'s" | b"\'S" | b"\'t" | b"\'T" | b"\'m" | b"\'M" | b"\'d" | b"\'D" |
                        b"\'re" | b"\'RE" | b"\'ve" | b"\'VE" | b"\'ll" | b"\'LL"
                    );

                    // --- NEW GUARD 2: don't merge if inner content length is exactly 1 ---
                    // (matches the slow tokenizer behavior seen for `"x"`/`'y'`)
                    let mut inner = cur_stripped[quote.len_utf8()..].chars();
                    let one = inner.next();
                    let two = inner.next(); // second char (if any)
                    let is_single_inner_alnum =
                        one.is_some() && two.is_none() && one.unwrap().is_alphanumeric();

                    if !is_bare_contr && !is_single_inner_alnum {
                        // merge: extend current token to include the closing quote
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
/// Uses native multi-pass approach entirely in indices space for maximum efficiency
pub fn pretokenize_fast_indices(text: &str) -> Vec<usize> {
    let initial = pretokenize_fast_single_pass_indices(text);
    let split_ws = split_mixed_whitespace_indices(text, &initial);
    let fixed   = fix_contractions_indices(text, &split_ws);
    let fused   = fuse_hspace_indices(text, &fixed);
    // Do NOT merge double quotes in indices mode to match slow tokenizer behavior
    let merged_single = merge_trailing_quote_indices(text, &fused, '\'');
    merged_single
}

/// Split any whitespace-only token that contains mixed types of horizontal whitespace
/// (e.g., space + tab, space + NBSP) into per-character tokens. Newlines are excluded.
fn split_mixed_whitespace_indices(text: &str, end_indices: &[usize]) -> Vec<usize> {
    if end_indices.is_empty() { return Vec::new(); }
    let mut out: Vec<usize> = Vec::with_capacity(end_indices.len());
    let mut prev_end = 0usize;
    for &end in end_indices {
        let tok = &text[prev_end..end];
        let is_ws_no_nl = !tok.contains('\n') && !tok.contains('\r') && tok.chars().all(|c| c.is_whitespace());
        if is_ws_no_nl {
            // If whitespace token mixes different kinds (e.g., ' ' + '\t' or ' ' + NBSP),
            // split off the final codepoint as its own token. This mirrors how the string
            // tokenizer handles donation using the last character while keeping the rest.
            let mut distinct: Option<char> = None;
            let mut mixed = false;
            let mut last_char: Option<char> = None;
            for ch in tok.chars() {
                last_char = Some(ch);
                if let Some(first) = distinct {
                    if ch != first { mixed = true; }
                } else {
                    distinct = Some(ch);
                }
            }

            if mixed {
                if let Some(last) = last_char {
                    let split_at = end - last.len_utf8();
                    // Push the prefix (all but last char), then the last char
                    if split_at > prev_end { out.push(split_at); }
                    out.push(end);
                } else {
                    out.push(end);
                }
            } else {
                out.push(end);
            }
        } else {
            out.push(end);
        }
        prev_end = end;
    }
    out
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

    #[test]
    fn test_space_punct_letters_split() {
        // Test space + punct + letters cases
        let text = " <Select";
        let indices = pretokenize_fast_indices(text);
        let strings = indices_to_strings(text, &indices);
        assert_eq!(strings, vec![" <", "Select"]);

        let text = " (options";
        let indices = pretokenize_fast_indices(text);
        let strings = indices_to_strings(text, &indices);
        assert_eq!(strings, vec![" (", "options"]);

        let text = " @NotNull";
        let indices = pretokenize_fast_indices(text);
        let strings = indices_to_strings(text, &indices);
        assert_eq!(strings, vec![" @", "NotNull"]);

        let text = " .filter";
        let indices = pretokenize_fast_indices(text);
        let strings = indices_to_strings(text, &indices);
        assert_eq!(strings, vec![" .", "filter"]);

        let text = " \"target";
        let indices = pretokenize_fast_indices(text);
        let strings = indices_to_strings(text, &indices);
        assert_eq!(strings, vec![" \"", "target"]);
    }

    #[test]
    fn test_quote_pairing_in_code() {
        // Test that quotes don't pair inside code
        let text = "x=(b\"x\")";
        let indices = pretokenize_fast_indices(text);
        let strings = indices_to_strings(text, &indices);
        assert_eq!(strings, vec!["x", "=(", "b", "\"x", "\")"]);
    }

    #[test]
    fn test_tabs_before_digits() {
        // "\n\t\t\t9" → split tabs as (n-1) + 1, then digit
        let text = "\n\t\t\t9";
        let indices = pretokenize_fast_indices(text);
        let strings = indices_to_strings(text, &indices);
        assert_eq!(strings, vec!["\n", "\t\t", "\t", "9"]);
    }

    #[test]
    fn test_unicode_numerics() {
        // Test Unicode numerics (Roman numerals) get space donated like letters
        let text = "Final Fantasy Ⅻ";
        let indices = pretokenize_fast_indices(text);
        let strings = indices_to_strings(text, &indices);
        assert_eq!(strings, vec!["Final", " Fantasy", " Ⅻ"]);
    }

    #[test]
    fn test_quote_pairing_with_whitespace() {
        // Test quote pairing with preceding whitespace/tabs
        let text = " \"PerPixelAdjust\"";
        let indices = pretokenize_fast_indices(text);
        let strings = indices_to_strings(text, &indices);
        assert_eq!(strings, vec![" \"PerPixelAdjust\""]);

        let text = "\t\"id\" uuid";
        let indices = pretokenize_fast_indices(text);
        let strings = indices_to_strings(text, &indices);
        assert_eq!(strings, vec!["\t", "\"id\"", " uuid"]);
    }

    #[test]
    fn test_opening_quotes_with_letters() {
        // Test opening quotes followed by letters (not contractions)
        let text = "\t'TIMEOUT'_Get";
        let indices = pretokenize_fast_indices(text);
        let strings = indices_to_strings(text, &indices);
        assert_eq!(strings, vec!["\t", "'TIMEOUT", "'_Get"]);

        let text = "\t'serviceaccount'\t";
        let indices = pretokenize_fast_indices(text);
        let strings = indices_to_strings(text, &indices);
        assert_eq!(strings, vec!["\t", "'serviceaccount'", "\t"]);
    }

    #[test]
    fn test_tabs_with_double_digits() {
        // Test the specific case mentioned: "\t\t10" - split tabs before digits
        let text = "\t\t10";
        let indices = pretokenize_fast_indices(text);
        let strings = indices_to_strings(text, &indices);
        assert_eq!(strings, vec!["\t", "\t", "1", "0"]);
    }

    #[test]
    fn test_failing_edge_cases() {
        // Edge case 1: Space + punct + letters should split
        let text = " '<link";
        let indices = pretokenize_fast_indices(text);
        let strings = indices_to_strings(text, &indices);
        assert_eq!(strings, vec![" '<", "link"]);

        // Edge case 2: Tabs before digits should be consistent (split policy)
        let text = "\t\t\t\t1";
        let indices = pretokenize_fast_indices(text);
        let strings = indices_to_strings(text, &indices);
        assert_eq!(strings, vec!["\t\t\t", "\t", "1"]);

        // Edge case 3: Quote pairing should work
        let text = "\"Register\"";
        let indices = pretokenize_fast_indices(text);
        let strings = indices_to_strings(text, &indices);
        assert_eq!(strings, vec!["\"Register\""]);

        // Edge case 4: Escape sequences should be handled properly
        let text = " '\\n";
        let indices = pretokenize_fast_indices(text);
        let strings = indices_to_strings(text, &indices);
        assert_eq!(strings, vec![" '\\n"]);
    }
}