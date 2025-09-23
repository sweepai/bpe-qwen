use crate::pretokenization::GLOBAL_QWEN_FAST_REGEX;
use unicode_general_category::{get_general_category, GeneralCategory};

/// Determine if a character is horizontal whitespace excluding CR and LF
#[inline]
fn is_hspace_no_crlf(c: char) -> bool {
    c.is_whitespace() && c != '\r' && c != '\n'
}

/// Punctuation for our purposes: not whitespace and not alphanumeric
#[inline]
fn is_punct(c: char) -> bool { !c.is_whitespace() && !is_letter(c) && !c.is_numeric() }

#[inline(always)]
fn is_letter(c: char) -> bool {
    matches!(get_general_category(c),
        GeneralCategory::UppercaseLetter |
        GeneralCategory::LowercaseLetter |
        GeneralCategory::TitlecaseLetter |
        GeneralCategory::ModifierLetter |
        GeneralCategory::OtherLetter)
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum CharClass { Ws, Letter, Numeric, Other }

#[inline(always)]
fn classify(c: char) -> CharClass {
    // ASCII fast-path
    if c.is_ascii() {
        if c.is_ascii_whitespace() { return CharClass::Ws; }
        if c.is_ascii_alphabetic() { return CharClass::Letter; }
        if c.is_ascii_digit() { return CharClass::Numeric; }
        return CharClass::Other;
    }
    if c.is_whitespace() { return CharClass::Ws; }
    // Match the regex \p{L} exactly using general category (faster than hashmap caching here)
    if matches!(get_general_category(c),
        GeneralCategory::UppercaseLetter |
        GeneralCategory::LowercaseLetter |
        GeneralCategory::TitlecaseLetter |
        GeneralCategory::ModifierLetter |
        GeneralCategory::OtherLetter) {
        return CharClass::Letter;
    }
    if c.is_numeric() { return CharClass::Numeric; }
    CharClass::Other
}

#[inline(always)]
fn consume_ascii_letters(bytes: &[u8], mut j: usize, len: usize) -> usize {
    // Fast path: advance over consecutive ASCII letters [A-Za-z]
    while j < len {
        // SAFETY: bounds checked by loop condition; using unchecked avoids redundant bounds checks
        let b = unsafe { *bytes.get_unchecked(j) };
        if b < 0x80 {
            let lower = b | 0x20;
            if lower >= b'a' && lower <= b'z' { j += 1; continue; }
        }
        break;
    }
    j
}

/// Case-insensitive check for a contraction token starting at byte position `pos`.
/// Matches one of: 's, 't, 'm, 'd, 're, 've, 'll (ASCII-insensitive)
#[inline(always)]
fn contraction_len(bytes: &[u8], pos: usize) -> Option<usize> {
    if pos >= bytes.len() || bytes[pos] != b'\'' { return None; }
    let b1 = bytes.get(pos + 1).copied();
    let b2 = bytes.get(pos + 2).copied();
    match (b1, b2) {
        // 2-byte contractions
        (Some(b), _) if matches!(b | 0x20, b's' | b't' | b'm' | b'd') => Some(2),
        // 3-byte: re, ve, ll
        (Some(b1), Some(b2)) => {
            let c1 = b1 | 0x20; let c2 = b2 | 0x20;
            if (c1 == b'r' && c2 == b'e') || (c1 == b'v' && c2 == b'e') || (c1 == b'l' && c2 == b'l') {
                Some(3)
            } else { None }
        }
        _ => None,
    }
}

#[inline(always)]
fn next_char_at_ascii_fast(s: &str, _bytes: &[u8], pos: usize) -> Option<(char, usize)> {
    // Re-enable ASCII fast path: vastly reduces overhead in tight inner loops.
    let bytes = s.as_bytes();
    if pos >= bytes.len() { return None; }
    // SAFETY: bounds checked just above; get_unchecked avoids redundant checks in hot path.
    let b = unsafe { *bytes.get_unchecked(pos) };
    if b < 0x80 {
        return Some((b as char, pos + 1));
    }
    s.get(pos..).and_then(|tail| tail.chars().next().map(|ch| {
        let w = ch.len_utf8();
        (ch, pos + w)
    }))
}

/// New: automaton implementation of QWEN_PATTERN_FAST that returns end indices only
pub fn pretokenize_fast_single_pass_indices_automaton(text: &str) -> Vec<usize> {
    let mut ends: Vec<usize> = Vec::with_capacity(text.len() / 4 + 8);
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut pos: usize = 0;

    while pos < len {
        // 1) Contractions: (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if let Some(clen) = contraction_len(bytes, pos) {
            ends.push(pos + clen);
            pos += clen;
            continue;
        }

        // Current character
        let (ch, next_pos1) = match next_char_at_ascii_fast(text, bytes, pos) { Some(t) => t, None => break };
        let class_ch = classify(ch);

        // 2) [^\S\r\n]\p{L}+  (one hspace-not-CRLF + letters)
        if is_hspace_no_crlf(ch) {
            if let Some((nch, mut j)) = next_char_at_ascii_fast(text, bytes, next_pos1) {
                if classify(nch) == CharClass::Letter {
                    while j < len {
                        if let Some((c2, j2)) = next_char_at_ascii_fast(text, bytes, j) {
                            if classify(c2) == CharClass::Letter { j = j2; } else { break; }
                        } else { break; }
                    }
                    ends.push(j);
                    pos = j;
                    continue;
                }
            }
        }

        // 3) [^\s\p{L}\p{N}]?\p{L}+
        if class_ch == CharClass::Letter {
            let mut j = next_pos1;
            // ASCII-letter hot loop
            j = consume_ascii_letters(bytes, j, len);
            while j < len {
                if let Some((c2, j2)) = next_char_at_ascii_fast(text, bytes, j) {
                    if classify(c2) == CharClass::Letter { j = j2; j = consume_ascii_letters(bytes, j, len); } else { break; }
                } else { break; }
            }
            ends.push(j);
            pos = j;
            continue;
        } else if class_ch == CharClass::Other {
            if let Some((nch, mut j)) = next_char_at_ascii_fast(text, bytes, next_pos1) {
                if classify(nch) == CharClass::Letter {
                    // ASCII-letter hot loop
                    j = consume_ascii_letters(bytes, j, len);
                    while j < len {
                        if let Some((c2, j2)) = next_char_at_ascii_fast(text, bytes, j) {
                            if classify(c2) == CharClass::Letter { j = j2; j = consume_ascii_letters(bytes, j, len); } else { break; }
                        } else { break; }
                    }
                    ends.push(j);
                    pos = j;
                    continue;
                }
            }
        }

        // 4) \p{N}  (single numeric codepoint)
        if ch.is_numeric() {
            ends.push(next_pos1);
            pos = next_pos1;
            continue;
        }

        // 5)  ?[^\s\p{L}\p{N}]+[\r\n]*
        if ch == ' ' {
            if let Some((nch, mut j)) = next_char_at_ascii_fast(text, bytes, next_pos1) {
                if classify(nch) == CharClass::Other {
                    while j < len {
                        if let Some((c2, j2)) = next_char_at_ascii_fast(text, bytes, j) {
                            if classify(c2) == CharClass::Other { j = j2; } else { break; }
                        } else { break; }
                    }
                    while j < len {
                        if let Some((c2, j2)) = next_char_at_ascii_fast(text, bytes, j) {
                            if c2 == '\r' || c2 == '\n' { j = j2; } else { break; }
                        } else { break; }
                    }
                    ends.push(j);
                    pos = j;
                    continue;
                }
            }
        }
        if class_ch == CharClass::Other {
            let mut j = next_pos1;
            while j < len {
                if let Some((c2, j2)) = next_char_at_ascii_fast(text, bytes, j) {
                    if classify(c2) == CharClass::Other { j = j2; } else { break; }
                } else { break; }
            }
            while j < len {
                if let Some((c2, j2)) = next_char_at_ascii_fast(text, bytes, j) {
                    if c2 == '\r' || c2 == '\n' { j = j2; } else { break; }
                } else { break; }
            }
            ends.push(j);
            pos = j;
            continue;
        }

        // 6) \s*[\r\n]+ - zero or more whitespace followed by one or more newlines
        if ch == '\r' || ch == '\n' {
            let mut j = pos;

            // This implements the full \s*[\r\n]+ pattern
            // The pattern is: zero or more whitespace, then one or more newlines
            // We need to handle sequences like "\n    \n" as a single token
            // But NOT consume trailing whitespace that doesn't lead to newlines

            // First consume initial newlines
            while j < len {
                if let Some((c2, j2)) = next_char_at_ascii_fast(text, bytes, j) {
                    if c2 == '\r' || c2 == '\n' {
                        j = j2;
                    } else {
                        break;
                    }
                } else { break; }
            }

            // Now look for the pattern: whitespace followed by more newlines
            // This handles cases like "\n    \n" where we want the whole thing as one token
            loop {
                let whitespace_start = j;
                // Consume whitespace (not CR/LF)
                while j < len {
                    if let Some((c2, j2)) = next_char_at_ascii_fast(text, bytes, j) {
                        if c2.is_whitespace() && c2 != '\r' && c2 != '\n' {
                            j = j2;
                        } else {
                            break;
                        }
                    } else { break; }
                }

                // Check if we have newlines after the whitespace
                let newline_start = j;
                while j < len {
                    if let Some((c2, j2)) = next_char_at_ascii_fast(text, bytes, j) {
                        if c2 == '\r' || c2 == '\n' {
                            j = j2;
                        } else {
                            break;
                        }
                    } else { break; }
                }

                // If we didn't find any newlines after the whitespace, backtrack
                if j == newline_start {
                    j = whitespace_start;
                    break;
                }
                // Otherwise continue the loop to look for more whitespace+newlines
            }

            ends.push(j);
            pos = j;
            continue;
        }

        // Check for whitespace that precedes newlines (part of \s*[\r\n]+ pattern)
        if ch.is_whitespace() && ch != '\r' && ch != '\n' {
            // Look ahead to see if this whitespace run leads to a newline
            let mut j = pos;
            let mut found_newline = false;

            // Scan ahead through whitespace to see if we hit a newline
            while j < len {
                if let Some((c2, j2)) = next_char_at_ascii_fast(text, bytes, j) {
                    if c2.is_whitespace() && c2 != '\r' && c2 != '\n' {
                        j = j2;
                    } else if c2 == '\r' || c2 == '\n' {
                        found_newline = true;
                        break;
                    } else {
                        break;
                    }
                } else { break; }
            }

            if found_newline {
                // This is whitespace that precedes newlines - consume as \s*[\r\n]+
                // Use the same logic as the newline-first case
                j = pos; // Reset to start of whitespace

                // Consume all the initial whitespace
                while j < len {
                    if let Some((c2, j2)) = next_char_at_ascii_fast(text, bytes, j) {
                        if c2.is_whitespace() && c2 != '\r' && c2 != '\n' {
                            j = j2;
                        } else {
                            break;
                        }
                    } else { break; }
                }

                // Now look for the pattern: newlines followed by more whitespace+newlines
                loop {
                    // Consume newlines
                    let newline_start = j;
                    while j < len {
                        if let Some((c2, j2)) = next_char_at_ascii_fast(text, bytes, j) {
                            if c2 == '\r' || c2 == '\n' {
                                j = j2;
                            } else {
                                break;
                            }
                        } else { break; }
                    }

                    // If we didn't consume any newlines, we're done
                    if j == newline_start {
                        break;
                    }

                    // Now consume any whitespace (not CR/LF)
                    let whitespace_start = j;
                    while j < len {
                        if let Some((c2, j2)) = next_char_at_ascii_fast(text, bytes, j) {
                            if c2.is_whitespace() && c2 != '\r' && c2 != '\n' {
                                j = j2;
                            } else {
                                break;
                            }
                        } else { break; }
                    }

                    // Check if we have newlines after the whitespace
                    let peek_j = j;
                    let mut has_more_newlines = false;
                    let peek_pos = peek_j;
                    while peek_pos < len {
                        if let Some((c2, _)) = next_char_at_ascii_fast(text, bytes, peek_pos) {
                            if c2 == '\r' || c2 == '\n' {
                                has_more_newlines = true;
                                break;
                            } else {
                                break;
                            }
                        } else { break; }
                    }

                    // If no more newlines after this whitespace, backtrack and stop
                    if !has_more_newlines {
                        j = whitespace_start;
                        break;
                    }
                    // Otherwise continue the loop to consume more newlines
                }
                ends.push(j);
                pos = j;
                continue;
            }
        }

        // 7/8) \s+ (and \s+\z) for non-CR/LF whitespace runs
        if ch.is_whitespace() && ch != '\r' && ch != '\n' {
            let mut j = next_pos1;
            while j < len {
                if let Some((c2, j2)) = next_char_at_ascii_fast(text, bytes, j) {
                    if c2.is_whitespace() && c2 != '\r' && c2 != '\n' { j = j2; } else { break; }
                } else { break; }
            }
            ends.push(j);
            pos = j;
            continue;
        }

        // Safety net: consume current char as a token to avoid infinite loops
        ends.push(next_pos1);
        pos = next_pos1;
    }

    ends
}

/// Single-pass fast implementation using \z anchor instead of lookahead (indices version)
pub fn pretokenize_fast_single_pass_indices(text: &str) -> Vec<usize> {
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
    text[start..end].chars().next().map(|c| is_letter(c)).unwrap_or(false)
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
    if first_char.is_whitespace() || is_letter(first_char) || first_char.is_numeric() {
        return None;
    }

    let rest = chars.as_str();

    // Check if the rest starts with letters
    if rest.chars().next().map(|c| is_letter(c)).unwrap_or(false) {
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
            // For donation into non-ASCII numerics (e.g., ①, Ⅻ), require previous token to end with an alnum
            // to avoid donating after punctuation like '#'.
            let prev_is_alnum_last = if i == 0 { false } else { is_alnum_last_indices(text, end_indices, i - 1) };

            // CASE A: letters OR non-ASCII numerics (e.g., Ⅻ) -> donate one trailing hspace
            // For non-ASCII numerics, only donate if the previous token ends with alnum
            if next_starts_letter || (next_starts_non_ascii_numeric && prev_is_alnum_last) {
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
    // Extra pass: split runs of non-ASCII horizontal whitespace (e.g., U+2003 EM SPACE)
    // into per-codepoint tokens, while preserving ASCII space/tab runs intact.
    let split_non_ascii_ws = split_non_ascii_hspace_runs_indices(text, &split_ws);
    let fixed   = fix_contractions_indices(text, &split_non_ascii_ws);
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

/// Return true if char is horizontal whitespace excluding ASCII space and tab, and excluding newlines.
fn is_non_ascii_hspace(c: char) -> bool {
    c.is_whitespace() && c != ' ' && c != '\t' && c != '\n' && c != '\r'
}

fn is_thinish_space(c: char) -> bool {
    matches!(c,
        '\u{2009}' /* THIN SPACE */ |
        '\u{200A}' /* HAIR SPACE */ |
        '\u{202F}' /* NARROW NO-BREAK SPACE */ |
        '\u{205F}' /* MEDIUM MATHEMATICAL SPACE */
    )
}

/// New pass: split tokens that are composed entirely of non-ASCII horizontal whitespace
/// (e.g., NBSP, EM SPACE U+2003) into per-codepoint tokens. Leave ASCII spaces/tabs intact.
fn split_non_ascii_hspace_runs_indices(text: &str, end_indices: &[usize]) -> Vec<usize> {
    // Coalesce consecutive tokens that are entirely non-ASCII horizontal whitespace
    // into a single token. Leave ASCII space/tab runs and all other tokens intact.
    if end_indices.is_empty() { return Vec::new(); }
    let mut out: Vec<usize> = Vec::with_capacity(end_indices.len());
    let mut i = 0usize;
    let mut prev_end = 0usize;
    while i < end_indices.len() {
        let end = end_indices[i];
        let tok = &text[prev_end..end];
        let is_ws_no_nl = !tok.contains('\n') && !tok.contains('\r') && tok.chars().all(|c| c.is_whitespace());
        let all_non_ascii_hspace = is_ws_no_nl && tok.chars().all(is_non_ascii_hspace);

        if all_non_ascii_hspace {
            // Extend through subsequent tokens that are also all non-ASCII hspace
            let mut j = i + 1;
            let mut last_end = end;
            let run_start = prev_end;
            let mut run_ends: Vec<usize> = vec![end];
            while j < end_indices.len() {
                let next_end = end_indices[j];
                let next_tok = &text[last_end..next_end];
                let is_ws_no_nl_next = !next_tok.contains('\n') && !next_tok.contains('\r')
                    && next_tok.chars().all(|c| c.is_whitespace());
                let all_non_ascii_next = is_ws_no_nl_next && next_tok.chars().all(is_non_ascii_hspace);
                if all_non_ascii_next {
                    last_end = next_end;
                    run_ends.push(next_end);
                    j += 1;
                } else {
                    break;
                }
            }
            // Decide how to represent this run according to slow tokenizer behavior
            let run_slice = &text[run_start..last_end];
            let mut uniq: Vec<char> = Vec::new();
            for ch in run_slice.chars() {
                if !uniq.contains(&ch) { uniq.push(ch); }
                if uniq.len() > 8 { break; } // small cap; not expected to be large
            }

            if uniq.len() == 1 {
                let ch = uniq[0];
                if ch == '\u{2003}' { // EM SPACE: keep repeated EM SPACE as separate tokens
                    let mut cur = run_start;
                    for ch in run_slice.chars() {
                        cur += ch.len_utf8();
                        out.push(cur);
                    }
                } else if ch == '\u{00A0}' { // NBSP: coalesce repeated NBSP into one token
                    out.push(last_end);
                } else if is_thinish_space(ch) {
                    // Thin-ish space repeated: coalesce
                    out.push(last_end);
                } else {
                    // Default for other non-ASCII spaces: preserve original tokenization
                    out.extend(run_ends.into_iter());
                }
            } else {
                // Mixed different codepoints: coalesce only if all are thin-ish; otherwise preserve
                let all_thin = run_slice.chars().all(is_thinish_space);
                if all_thin {
                    out.push(last_end);
                } else {
                    out.extend(run_ends.into_iter());
                }
            }
            // Advance i and prev_end
            i = j;
            prev_end = last_end;
            continue;
        }

        // Default: keep token as-is
        out.push(end);
        prev_end = end;
        i += 1;
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
    fn test_debug_specific_failing_case() {
        let text = "    \n    \n";
        let re = regex::Regex::new(crate::pretokenization::QWEN_PATTERN_FAST).unwrap();

        println!("Text: {:?}", text);
        println!("Text bytes: {:?}", text.as_bytes());

        let expected: Vec<usize> = re.find_iter(text).map(|m| m.end()).collect();
        let actual = pretokenize_fast_single_pass_indices_automaton(text);

        println!("Expected: {:?}", expected);
        println!("Actual:   {:?}", actual);

        // Print the tokens for debugging
        let expected_tokens: Vec<&str> = {
            let mut tokens = Vec::new();
            let mut start = 0;
            for &end in &expected {
                tokens.push(&text[start..end]);
                start = end;
            }
            tokens
        };

        let actual_tokens: Vec<&str> = {
            let mut tokens = Vec::new();
            let mut start = 0;
            for &end in &actual {
                tokens.push(&text[start..end]);
                start = end;
            }
            tokens
        };

        println!("Expected tokens: {:?}", expected_tokens);
        println!("Actual tokens:   {:?}", actual_tokens);

        // Find the first difference
        for (i, (&exp, &act)) in expected.iter().zip(actual.iter()).enumerate() {
            if exp != act {
                println!("First difference at index {}: expected {}, got {}", i, exp, act);
                break;
            }
        }

        assert_eq!(actual, expected, "mismatch for {:?}", text);
    }

    #[test]
    fn test_automaton_single_pass_matches_regex_on_samples() {
        let samples = vec![
            "Hello, world!",
            "I've got 2 apples",
            "  tabs\tand spaces",
            "line1\r\nline2\nline3\rline4",
            " @NotNull <Select .filter \"target",
            "'re 've 'm 'll 'd 's 't",
            "中文 空格 test",
            " \u{00A0}\u{2003}mix",
            // Add some failing cases from the SFT test
            "formatType) {\n          case 'PQ':\n            \n            types.push({ type: '128k', size })\n     ",
            "\n  // shallow clone\n  ret.memo = memo.slice()\n  \n\n  return (cache[index] = ret)\n}\n\nexport function i",
            " test_sport_with_ancient_and_unusual_allowed\n    \n  end\n\n  def test_summer_olympics\n    assert @test",
            "\n      this.reloadAt = new AtomicLong();\n   }\n   \n   public int getTotalConnections()\n   {\n      if ",
            "space\")\npublic class NamespaceControllerV2 {\n    \n    private final NamespaceOperationService namesp",
            // Specific whitespace patterns that were failing
            "    \n    \n",
            "        \n        \n",
            " \n    \n",
            "d:\n    \n  ",
            "        \n",
            " \n",
        ];

        let re = regex::Regex::new(crate::pretokenization::QWEN_PATTERN_FAST).unwrap();
        for text in samples {
            let expected: Vec<usize> = re.find_iter(text).map(|m| m.end()).collect();
            let actual = pretokenize_fast_single_pass_indices_automaton(text);
            assert_eq!(actual, expected, "mismatch for {:?}\nexpected: {:?}\nactual:   {:?}", text, expected, actual);
        }
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