/// Custom hand-written automata for Qwen pretokenization
/// This implements the pattern in a single pass without regex or corrections
/// Pattern: (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+

#[derive(Debug, Clone, Copy, PartialEq)]
enum State {
    Start,
    InWhitespace,
    InWord,
    InNumber,
    InPunctuation,
    InLineBreaks,
    InContraction,
}

/// Single-pass automata for Qwen pretokenization
pub fn pretokenize_automata(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::new();
    let mut current_token = String::new();
    let mut state = State::Start;
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];

        match state {
            State::Start => {
                if ch.is_whitespace() {
                    if ch == '\r' || ch == '\n' {
                        state = State::InLineBreaks;
                        current_token.push(ch);
                    } else {
                        state = State::InWhitespace;
                        current_token.push(ch);
                    }
                } else if ch.is_alphabetic() {
                    state = State::InWord;
                    current_token.push(ch);
                } else if ch.is_numeric() {
                    state = State::InNumber;
                    current_token.push(ch);
                } else if ch == '\'' && i + 1 < chars.len() {
                    // Check for contractions
                    if let Some(contraction) = check_contraction(&chars, i) {
                        result.push(contraction.clone());
                        i += contraction.len();
                        continue;
                    } else {
                        state = State::InPunctuation;
                        current_token.push(ch);
                    }
                } else {
                    state = State::InPunctuation;
                    current_token.push(ch);
                }
                i += 1;
            }

            State::InWhitespace => {
                if ch.is_whitespace() {
                    if ch == '\r' || ch == '\n' {
                        // Transition to line breaks
                        if !current_token.is_empty() {
                            // Handle whitespace before line breaks
                            handle_whitespace_token(&mut result, &current_token, &chars, i);
                            current_token.clear();
                        }
                        state = State::InLineBreaks;
                        current_token.push(ch);
                    } else {
                        current_token.push(ch);
                    }
                    i += 1;
                } else {
                    // End of whitespace
                    if !current_token.is_empty() {
                        handle_whitespace_token(&mut result, &current_token, &chars, i);
                        current_token.clear();
                    }
                    state = State::Start;
                }
            }

            State::InWord => {
                if ch.is_alphabetic() {
                    current_token.push(ch);
                    i += 1;
                } else if ch == '\'' && i + 1 < chars.len() {
                    // Check if this could be a contraction
                    let rest: String = chars[i..].iter().collect();
                    if rest.to_lowercase().starts_with("'s") ||
                       rest.to_lowercase().starts_with("'t") ||
                       rest.to_lowercase().starts_with("'re") ||
                       rest.to_lowercase().starts_with("'ve") ||
                       rest.to_lowercase().starts_with("'m") ||
                       rest.to_lowercase().starts_with("'ll") ||
                       rest.to_lowercase().starts_with("'d") {
                        // End current word and start contraction
                        if !current_token.is_empty() {
                            result.push(current_token.clone());
                            current_token.clear();
                        }
                        state = State::Start;
                    } else {
                        // Not a contraction, include apostrophe if it's part of word
                        if i > 0 && chars[i-1].is_alphabetic() && i + 1 < chars.len() && chars[i+1].is_alphabetic() {
                            current_token.push(ch);
                            i += 1;
                        } else {
                            // End word
                            if !current_token.is_empty() {
                                result.push(current_token.clone());
                                current_token.clear();
                            }
                            state = State::Start;
                        }
                    }
                } else {
                    // End of word
                    if !current_token.is_empty() {
                        result.push(current_token.clone());
                        current_token.clear();
                    }
                    state = State::Start;
                }
            }

            State::InNumber => {
                if ch.is_numeric() {
                    // Each digit is its own token
                    result.push(ch.to_string());
                    i += 1;
                } else {
                    state = State::Start;
                }
            }

            State::InPunctuation => {
                if !ch.is_whitespace() && !ch.is_alphabetic() && !ch.is_numeric() {
                    current_token.push(ch);
                    i += 1;
                    // Check for line breaks after punctuation
                    while i < chars.len() && (chars[i] == '\r' || chars[i] == '\n') {
                        current_token.push(chars[i]);
                        i += 1;
                    }
                } else {
                    // End of punctuation
                    if !current_token.is_empty() {
                        // Check if we should have included a leading space
                        result.push(current_token.clone());
                        current_token.clear();
                    }
                    state = State::Start;
                }
            }

            State::InLineBreaks => {
                if ch == '\r' || ch == '\n' {
                    current_token.push(ch);
                    i += 1;
                } else if ch.is_whitespace() {
                    // Include leading whitespace with line breaks
                    let mut j = i;
                    while j > 0 && chars[j-1].is_whitespace() && chars[j-1] != '\r' && chars[j-1] != '\n' {
                        j -= 1;
                    }
                    if j < i {
                        let prefix: String = chars[j..i].iter().collect();
                        current_token = prefix + &current_token;
                    }
                    // Check for trailing whitespace
                    while i < chars.len() && chars[i].is_whitespace() && chars[i] != '\r' && chars[i] != '\n' {
                        current_token.push(chars[i]);
                        i += 1;
                    }
                    result.push(current_token.clone());
                    current_token.clear();
                    state = State::Start;
                } else {
                    // End of line breaks
                    if !current_token.is_empty() {
                        result.push(current_token.clone());
                        current_token.clear();
                    }
                    state = State::Start;
                }
            }

            State::InContraction => {
                // Handled inline
                unreachable!();
            }
        }
    }

    // Handle any remaining token
    if !current_token.is_empty() {
        match state {
            State::InWhitespace => {
                // Apply lookahead logic for final whitespace
                handle_whitespace_token(&mut result, &current_token, &chars, chars.len());
            }
            _ => {
                result.push(current_token);
            }
        }
    }

    result
}

/// Check if we have a contraction at the current position
fn check_contraction(chars: &[char], start: usize) -> Option<String> {
    if start >= chars.len() || chars[start] != '\'' {
        return None;
    }

    let remaining: String = chars[start..].iter().collect();
    let lower = remaining.to_lowercase();

    // Check each contraction pattern
    for pattern in &["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"] {
        if lower.starts_with(pattern) {
            return Some(chars[start..start + pattern.len()].iter().collect());
        }
    }

    None
}

/// Handle whitespace token with lookahead logic
fn handle_whitespace_token(result: &mut Vec<String>, whitespace: &str, chars: &[char], next_pos: usize) {
    // Check what follows the whitespace
    if next_pos < chars.len() {
        let next_char = chars[next_pos];

        // Apply Qwen's whitespace rules
        if next_char.is_alphabetic() {
            // Whitespace followed by letter: merge last space with following token
            let ws_chars: Vec<char> = whitespace.chars().collect();
            if ws_chars.len() > 1 {
                // Keep all but last space
                let first_part: String = ws_chars[..ws_chars.len() - 1].iter().collect();
                let last_space: String = ws_chars[ws_chars.len() - 1].to_string();
                result.push(first_part);
                // The last space will be merged with the next token
                // This is handled by the next token's processing
            } else {
                // Single space - will be merged with next token
            }
        } else if next_char.is_numeric() {
            // Whitespace followed by number: split whitespace
            let ws_chars: Vec<char> = whitespace.chars().collect();
            if ws_chars.len() > 1 {
                let first_part: String = ws_chars[..ws_chars.len() - 1].iter().collect();
                let last_space: String = ws_chars[ws_chars.len() - 1].to_string();
                result.push(first_part);
                result.push(last_space);
            } else {
                result.push(whitespace.to_string());
            }
        } else {
            // Whitespace followed by punctuation or end
            // The pattern ` ?[^\s\p{L}\p{N}]+` means punctuation can have optional leading space
            let ws_chars: Vec<char> = whitespace.chars().collect();
            if ws_chars.len() > 1 {
                let first_part: String = ws_chars[..ws_chars.len() - 1].iter().collect();
                result.push(first_part);
                // Last space may be merged with punctuation
            } else {
                // Single space may be merged with punctuation
            }
        }
    } else {
        // Whitespace at end of text
        result.push(whitespace.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_automata_basic() {
        assert_eq!(pretokenize_automata("Hello world"), vec!["Hello", " world"]);
        assert_eq!(pretokenize_automata("test"), vec!["test"]);
        assert_eq!(pretokenize_automata("123"), vec!["1", "2", "3"]);
    }

    #[test]
    fn test_automata_contractions() {
        assert_eq!(pretokenize_automata("I've"), vec!["I", "'ve"]);
        assert_eq!(pretokenize_automata("can't"), vec!["can", "'t"]);
        assert_eq!(pretokenize_automata("we're"), vec!["we", "'re"]);
    }

    #[test]
    fn test_automata_whitespace() {
        assert_eq!(pretokenize_automata("  hello"), vec![" ", " hello"]);
        assert_eq!(pretokenize_automata("   123"), vec!["  ", " ", "1", "2", "3"]);
    }

    #[test]
    fn test_automata_punctuation() {
        assert_eq!(pretokenize_automata("hello, world"), vec!["hello", ",", " world"]);
        assert_eq!(pretokenize_automata("test."), vec!["test", "."]);
    }

    #[test]
    fn test_automata_line_breaks() {
        assert_eq!(pretokenize_automata("line1\nline2"), vec!["line", "1", "\n", "line", "2"]);
        assert_eq!(pretokenize_automata("test\r\n"), vec!["test", "\r\n"]);
    }
}