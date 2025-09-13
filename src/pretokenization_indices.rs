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

    // Second pass: adjust boundaries for whitespace handling
    let mut result = Vec::with_capacity(matches.len());
    let text_bytes = text.as_bytes();
    let mut i = 0;

    while i < matches.len() {
        let (start, end) = matches[i];
        let mat_slice = &text[start..end];

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
                            // Check for contractions
                            let next_lower = next_slice.to_lowercase();
                            if next_lower.starts_with("'s") || next_lower.starts_with("'t") ||
                               next_lower.starts_with("'re") || next_lower.starts_with("'ve") ||
                               next_lower.starts_with("'m") || next_lower.starts_with("'ll") ||
                               next_lower.starts_with("'d") {
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
                                // Check if it would form a problematic pattern
                                let has_letters = next_slice.chars().skip(1).any(|c| c.is_alphabetic());
                                if has_letters {
                                    let letter_part: String = next_slice.chars().skip(1).collect();
                                    if letter_part.to_lowercase().starts_with("ve") ||
                                       letter_part.to_lowercase().starts_with("re") ||
                                       letter_part.to_lowercase().starts_with("ll") ||
                                       letter_part.to_lowercase().starts_with("s") ||
                                       letter_part.to_lowercase().starts_with("t") ||
                                       letter_part.to_lowercase().starts_with("m") ||
                                       letter_part.to_lowercase().starts_with("d") {
                                        // Would match contraction - handle specially
                                        let space_chars: Vec<char> = mat_slice.chars().collect();
                                        if space_chars.len() > 1 {
                                            let split_pos = start + mat_slice.len() - mat_slice.chars().last().unwrap().len_utf8();
                                            result.push((start, split_pos));
                                            // Find where letters start in next token
                                            let mut letter_start_offset = 0;
                                            for ch in next_slice.chars() {
                                                if ch.is_alphabetic() {
                                                    break;
                                                }
                                                letter_start_offset += ch.len_utf8();
                                            }
                                            if letter_start_offset > 0 {
                                                result.push((split_pos, next_start + letter_start_offset));
                                                result.push((next_start + letter_start_offset, next_end));
                                            } else {
                                                result.push((split_pos, next_end));
                                            }
                                        } else {
                                            // Single space
                                            let mut letter_start_offset = 0;
                                            for ch in next_slice.chars() {
                                                if ch.is_alphabetic() {
                                                    break;
                                                }
                                                letter_start_offset += ch.len_utf8();
                                            }
                                            if letter_start_offset > 0 {
                                                result.push((start, next_start + letter_start_offset));
                                                result.push((next_start + letter_start_offset, next_end));
                                            } else {
                                                result.push((start, next_end));
                                            }
                                        }
                                        i += 2;
                                        continue;
                                    }
                                }
                            }
                        }

                        // Default punctuation handling
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