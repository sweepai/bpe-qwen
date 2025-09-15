// Re-uses the string-based implementation for consistency

/// Fast pretokenization that returns only end positions of tokens
/// Returns only end positions since start of next token = end of previous token
/// Uses the same four-pass approach as the string version for consistency
pub fn pretokenize_fast_indices(text: &str) -> Vec<usize> {
    // Use the string-based implementation to get the correct tokens
    let tokens = crate::pretokenization::pretokenize_fast(text);

    // Convert tokens to end indices
    strings_to_indices(text, &tokens)
}



/// Convert string tokens to end indices
fn strings_to_indices(_text: &str, tokens: &[String]) -> Vec<usize> {
    let mut result = Vec::with_capacity(tokens.len());
    let mut pos = 0;

    for token in tokens {
        pos += token.len();
        result.push(pos);
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