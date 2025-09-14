use regex::Regex;

/// Pretokenization that returns byte indices instead of string slices
/// This avoids string allocations and makes the second pass more efficient
pub fn pretokenize_fast_indices(text: &str) -> Vec<(usize, usize)> {
    // Compile the regex pattern without lookahead
    let pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+";
    let re = Regex::new(pattern).expect("Failed to compile regex");

    // Find all matches and collect their byte positions
    let matches: Vec<(usize, usize)> = re
        .find_iter(text)
        .map(|m| (m.start(), m.end()))
        .collect();

    if matches.is_empty() {
        return matches;
    }

    // Second pass: fix incorrectly split contractions and adjust whitespace
    let mut result = Vec::with_capacity(matches.len());
    let text_bytes = text.as_bytes();
    let mut i = 0;

    while i < matches.len() {
        let (start, end) = matches[i];
        let mat_slice = &text[start..end];

        // Check if this is a contraction pattern that might have incorrectly split a word
        if mat_slice.to_lowercase() == "'s" || mat_slice.to_lowercase() == "'t" ||
           mat_slice.to_lowercase() == "'re" || mat_slice.to_lowercase() == "'ve" ||
           mat_slice.to_lowercase() == "'m" || mat_slice.to_lowercase() == "'ll" ||
           mat_slice.to_lowercase() == "'d" {
            // Check if the next token starts with letters (indicating it was incorrectly split)
            if i + 1 < matches.len() {
                let (next_start, next_end) = matches[i + 1];
                let next_slice = &text[next_start..next_end];

                // If next token starts with a letter, this was likely an incorrect split
                if !next_slice.is_empty() && next_slice.chars().next().unwrap().is_alphabetic() {
                    // Check if there's something before the apostrophe
                    if start > 0 {
                        // Look at the previous character
                        let prev_char_end = start;
                        let mut prev_char_start = start.saturating_sub(1);
                        while prev_char_start > 0 && !text.is_char_boundary(prev_char_start) {
                            prev_char_start -= 1;
                        }
                        let prev_char = &text[prev_char_start..prev_char_end];

                        // If previous character is not alphanumeric, this is a quoted word, not a contraction
                        if !prev_char.chars().next().map_or(false, |c| c.is_alphanumeric()) {
                            // Merge the incorrectly split tokens
                            result.push((start, next_end));
                            i += 2;
                            continue;
                        }
                    } else {
                        // If apostrophe is at the start, it's definitely not a contraction
                        result.push((start, next_end));
                        i += 2;
                        continue;
                    }
                }
            }
        }

        // Check if this match is only whitespace (but not containing \r or \n)
        if mat_slice.chars().all(|c| c.is_whitespace())
            && !mat_slice.contains('\r')
            && !mat_slice.contains('\n') {

            // Check what follows this match
            if i + 1 < matches.len() {
                let (next_start, next_end) = matches[i + 1];
                let next_slice = &text[next_start..next_end];

                if !next_slice.is_empty() {
                    let first_char = next_slice.chars().next().unwrap();

                    // Handle different cases based on what follows
                    if first_char.is_alphabetic() {
                        // Merge last space with alphabetic token
                        let space_chars: Vec<char> = mat_slice.chars().collect();

                        if space_chars.len() > 1 {
                            // Split: keep all but last space
                            let split_pos = start + mat_slice.len() - mat_slice.chars().last().unwrap().len_utf8();
                            result.push((start, split_pos));
                            result.push((split_pos, next_end));
                            i += 2;
                            continue;
                        } else {
                            // Single space - merge with next
                            result.push((start, next_end));
                            i += 2;
                            continue;
                        }
                    } else if first_char.is_numeric() {
                        // For numbers, split whitespace but keep separate
                        let space_chars: Vec<char> = mat_slice.chars().collect();

                        if space_chars.len() > 1 {
                            let split_pos = start + mat_slice.len() - mat_slice.chars().last().unwrap().len_utf8();
                            result.push((start, split_pos));
                            result.push((split_pos, end));
                            i += 1;
                            continue;
                        }
                    } else if !first_char.is_whitespace() {
                        // Handle punctuation
                        if next_slice.starts_with('\'') && next_slice.len() > 1 {
                            // Check for actual contractions (must be exactly a contraction, not a quoted word)
                            let next_lower = next_slice.to_lowercase();
                            let is_contraction = next_lower == "'s" || next_lower == "'t" ||
                                                 next_lower == "'re" || next_lower == "'ve" ||
                                                 next_lower == "'m" || next_lower == "'ll" ||
                                                 next_lower == "'d";

                            if is_contraction {
                                // Don't merge with contractions
                                let space_chars: Vec<char> = mat_slice.chars().collect();
                                if space_chars.len() > 1 {
                                    let split_pos = start + mat_slice.len() - mat_slice.chars().last().unwrap().len_utf8();
                                    result.push((start, split_pos));
                                    result.push((split_pos, end));
                                    i += 1;
                                    continue;
                                }
                            } else {
                                // This is a quoted word, not a contraction
                                // Don't merge spaces with quoted words - keep them separate
                                let space_chars: Vec<char> = mat_slice.chars().collect();
                                if space_chars.len() > 1 {
                                    let split_pos = start + mat_slice.len() - mat_slice.chars().last().unwrap().len_utf8();
                                    result.push((start, split_pos));
                                    result.push((split_pos, end));
                                } else {
                                    result.push((start, end));
                                }
                                i += 1;
                                continue;
                            }
                        }

                        // Default punctuation handling (non-quote punctuation)
                        let space_chars: Vec<char> = mat_slice.chars().collect();
                        if space_chars.len() > 1 {
                            let split_pos = start + mat_slice.len() - mat_slice.chars().last().unwrap().len_utf8();
                            result.push((start, split_pos));
                            result.push((split_pos, next_end));
                            i += 2;
                            continue;
                        } else {
                            result.push((start, next_end));
                            i += 2;
                            continue;
                        }
                    }
                }
            }
        }

        // Default: keep the match as-is
        result.push((start, end));
        i += 1;
    }

    result
}

/// Convert indices to actual string slices for testing
pub fn indices_to_strings(text: &str, indices: &[(usize, usize)]) -> Vec<String> {
    indices.iter()
        .map(|&(start, end)| text[start..end].to_string())
        .collect()
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