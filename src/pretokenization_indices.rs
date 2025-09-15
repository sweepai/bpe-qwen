use crate::pretokenization::GLOBAL_QWEN_REGEX;

/// Fast pretokenization that returns only end positions of tokens
/// Returns only end positions since start of next token = end of previous token
/// Implements the same complex algorithm as pretokenization.rs but with indices only
pub fn pretokenize_fast_indices(text: &str) -> Vec<usize> {
    let regex = GLOBAL_QWEN_REGEX.get().expect("Global regex not initialized");

    // First pass: collect all regex matches
    let matches: Vec<_> = regex.find_iter(text).collect();

    // Second pass: fix incorrectly split contractions and apply whitespace correction
    let mut result = Vec::with_capacity(matches.len());
    let mut i = 0;

    while i < matches.len() {
        let mat = matches[i];
        let _start = mat.start();
        let end = mat.end();
        let mat_str = mat.as_str();

        // Note: Previously there was logic here to merge quoted strings like "context" into single tokens,
        // but this caused inconsistency with the string-based tokenization which keeps quotes separate.
        // The string implementation produces ['"context', '"'] not ['"context"'], so we removed the merging.

        // Check if this match is a contraction pattern that might have incorrectly split a word
        if mat_str == "'s" || mat_str == "'t" || mat_str == "'re" || mat_str == "'ve" ||
            mat_str == "'m" || mat_str == "'ll" || mat_str == "'d" ||
            mat_str == "'S" || mat_str == "'T" || mat_str == "'RE" || mat_str == "'VE" ||
            mat_str == "'M" || mat_str == "'LL" || mat_str == "'D" {

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
                            // Reconstruct what the last token was
                            let last_end = result[result.len() - 1];
                            let last_start = if result.len() == 1 { 0 } else { result[result.len() - 2] };
                            let last_token = &text[last_start..last_end];

                            if last_token == " " {
                                // Pop the single space and replace with space + quote
                                result.pop();
                                let space_quote_end = start_pos + "'".len();
                                result.push(space_quote_end);
                                // Add the full word (contraction pattern + letters)
                                result.push(matches[i + 1].end());
                                i += 2;
                                continue;
                            }
                        }

                        // Otherwise just merge the incorrectly split tokens
                        result.push(matches[i + 1].end());
                        i += 2;
                        continue;
                    }
                }
            }
        }

        // Check if this match is only whitespace (but not containing \r or \n)
        // \r and \n are handled specially by the regex and should not be split
        if mat_str.chars().all(|c| c.is_whitespace())
            && !mat_str.contains('\r')
            && !mat_str.contains('\n') {
            // Check what follows this match
            if i + 1 < matches.len() {
                let next = matches[i + 1].as_str();

                // Special case: For comment patterns like " *", keep space and asterisk together
                // This must be checked first before other punctuation logic
                if next == "*" {
                    let space_chars: Vec<char> = mat_str.chars().collect();
                    if space_chars.len() > 1 {
                        // Multiple spaces - split off all but last space, then merge last space with asterisk
                        let split_pos = end - mat_str.chars().last().unwrap().len_utf8();
                        result.push(split_pos);
                    }
                    // Merge the (last) space with the asterisk
                    result.push(matches[i + 1].end());
                    i += 2;
                    continue;
                }

                // The lookahead \s+(?!\S) means "whitespace NOT followed by non-whitespace"
                // So when whitespace IS followed by non-whitespace, we need to handle it specially
                if !next.is_empty() {
                    let first_char = next.chars().next().unwrap();

                    if first_char.is_alphabetic() {
                        // For alphabetic chars, merge the last space with the next token
                        let space_chars: Vec<char> = mat_str.chars().collect();

                        if space_chars.len() > 1 {
                            // Keep all but last space as separate token
                            let split_pos = end - mat_str.chars().last().unwrap().len_utf8();
                            result.push(split_pos);
                            // Merge last space with next token
                            result.push(matches[i + 1].end());
                            i += 2; // Skip next token since we merged it
                            continue;
                        } else {
                            // Single space - merge with next token
                            result.push(matches[i + 1].end());
                            i += 2; // Skip next token since we merged it
                            continue;
                        }
                    } else if first_char.is_numeric() {
                        // For numbers, split whitespace but keep them separate
                        let space_chars: Vec<char> = mat_str.chars().collect();

                        if space_chars.len() > 1 {
                            // Split into n-1 spaces and 1 space (like slow version)
                            let split_pos = end - mat_str.chars().last().unwrap().len_utf8();
                            result.push(split_pos);
                            result.push(end);
                            i += 1; // Continue to process next token normally
                            continue;
                        }
                    } else if !first_char.is_whitespace() {
                        // For punctuation/other non-whitespace, non-alphabetic, non-numeric
                        // The pattern ` ?[^\s\p{L}\p{N}]+` matches space + punctuation
                        // But we need to be careful not to create tokens that would be split by contractions

                        // Special case: Don't merge tab characters with following punctuation
                        // Tab should always remain separate from punctuation like @, ", ., etc.
                        if mat_str.contains('\t') {
                            // Keep tabs separate from punctuation
                            let space_chars: Vec<char> = mat_str.chars().collect();
                            if space_chars.len() > 1 {
                                let split_pos = end - mat_str.chars().last().unwrap().len_utf8();
                                result.push(split_pos);
                                result.push(end);
                                i += 1;
                                continue;
                            } else {
                                // Single tab - keep separate
                                result.push(end);
                                i += 1;
                                continue;
                            }
                        }

                        // Special case: For comment patterns like " *", keep space and asterisk together
                        // This matches patterns like JavaDoc comment continuations
                        if next.starts_with('*') && next.len() == 1 {
                            // Current token is whitespace, next token is a single asterisk
                            let space_chars: Vec<char> = mat_str.chars().collect();
                            if space_chars.len() > 1 {
                                // Multiple spaces - split off all but last space, then merge last space with asterisk
                                let split_pos = end - mat_str.chars().last().unwrap().len_utf8();
                                result.push(split_pos);
                            }
                            // Merge the (last) space with the asterisk
                            result.push(matches[i + 1].end());
                            i += 2;
                            continue;
                        }

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
                                let space_chars: Vec<char> = mat_str.chars().collect();
                                if space_chars.len() > 1 {
                                    let split_pos = end - mat_str.chars().last().unwrap().len_utf8();
                                    result.push(split_pos);
                                    result.push(end);
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
                                        let space_chars: Vec<char> = mat_str.chars().collect();
                                        if space_chars.len() > 1 {
                                            let split_pos = end - mat_str.chars().last().unwrap().len_utf8();
                                            result.push(split_pos);
                                            // Merge space with just the punctuation, not the letters
                                            let punct_end = matches[i + 1].start() + "'".len();
                                            result.push(punct_end);
                                            // Add the letters separately
                                            result.push(matches[i + 1].end());
                                            i += 2;
                                            continue;
                                        } else {
                                            // Single space - merge with punctuation only
                                            let punct_end = matches[i + 1].start() + "'".len();
                                            result.push(punct_end);
                                            result.push(matches[i + 1].end());
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
                            let space_chars: Vec<char> = mat_str.chars().collect();

                            if space_chars.len() > 1 {
                                let split_pos = end - mat_str.chars().last().unwrap().len_utf8();
                                result.push(split_pos);

                                // Split punctuation from letters
                                let letter_start = next_chars.iter().position(|c| c.is_alphabetic()).unwrap_or(next_chars.len());
                                if letter_start > 0 && letter_start < next_chars.len() {
                                    let punct_end = matches[i + 1].start() + next_chars[..letter_start].iter().map(|c| c.len_utf8()).sum::<usize>();
                                    result.push(punct_end);
                                    result.push(matches[i + 1].end());
                                } else {
                                    result.push(matches[i + 1].end());
                                }
                                i += 2;
                                continue;
                            } else {
                                // Single space
                                let letter_start = next_chars.iter().position(|c| c.is_alphabetic()).unwrap_or(next_chars.len());
                                if letter_start > 0 && letter_start < next_chars.len() {
                                    let punct_end = matches[i + 1].start() + next_chars[..letter_start].iter().map(|c| c.len_utf8()).sum::<usize>();
                                    result.push(punct_end);
                                    result.push(matches[i + 1].end());
                                } else {
                                    result.push(matches[i + 1].end());
                                }
                                i += 2;
                                continue;
                            }
                        } else if !next.starts_with(' ') {
                            // Pure punctuation without letters
                            let space_chars: Vec<char> = mat_str.chars().collect();

                            if space_chars.len() > 1 {
                                let split_pos = end - mat_str.chars().last().unwrap().len_utf8();
                                result.push(split_pos);
                                result.push(matches[i + 1].end());
                                i += 2;
                                continue;
                            } else {
                                result.push(matches[i + 1].end());
                                i += 2;
                                continue;
                            }
                        }
                    }
                }
            }
        }

        // Default case: keep the match as-is
        result.push(end);
        i += 1;
    }

    result
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