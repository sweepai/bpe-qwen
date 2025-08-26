use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::Deserialize;
use std::collections::HashMap;
use regex::Regex;
use unicode_normalization::UnicodeNormalization;

use bpe::byte_pair_encoding::BytePairEncoding;

/// Represents a HuggingFace tokenizer configuration
#[derive(Debug, Deserialize)]
struct HFTokenizerConfig {
    model: HFModel,
    pre_tokenizer: Option<HFPreTokenizer>,
    normalizer: Option<HFNormalizer>,
    added_tokens: Option<Vec<AddedToken>>,
}

#[derive(Debug, Deserialize)]
struct HFModel {
    #[serde(rename = "type")]
    _model_type: String,
    vocab: HashMap<String, u32>,
    merges: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct HFPreTokenizer {
    #[serde(rename = "type")]
    _tokenizer_type: String,
    pretokenizers: Option<Vec<PreTokenizer>>,
}

#[derive(Debug, Deserialize)]
struct PreTokenizer {
    #[serde(rename = "type")]
    _tokenizer_type: String,
    pattern: Option<PatternConfig>,
    #[allow(dead_code)]
    behavior: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PatternConfig {
    #[serde(rename = "Regex")]
    regex: Option<String>,
}

#[derive(Debug, Deserialize)]
struct HFNormalizer {
    #[serde(rename = "type")]
    normalizer_type: String,
}

#[derive(Debug, Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
    special: bool,
}

/// QwenTokenizer - A fast BPE tokenizer for Qwen models
#[pyclass]
struct QwenTokenizer {
    bpe: BytePairEncoding,
    pre_tokenizer_regex: Option<Regex>,
    normalizer_type: Option<String>,
    special_tokens: HashMap<String, u32>,
    special_token_ids: HashMap<u32, String>,
    token_id_map: HashMap<u32, u32>,  // Maps original token IDs to deduplicated indices
    reverse_token_id_map: HashMap<u32, u32>,  // Maps deduplicated indices back to original IDs
}

#[pymethods]
impl QwenTokenizer {
    /// Create a new QwenTokenizer from vocab.json and merges.txt files
    #[new]
    fn new(vocab_path: &str, merges_path: &str) -> PyResult<Self> {
        // Read vocab.json
        let vocab_content = std::fs::read_to_string(vocab_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("Failed to read vocab file: {}", e)
            ))?;
        
        let vocab: HashMap<String, u32> = serde_json::from_str(&vocab_content)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Failed to parse vocab: {}", e)
            ))?;

        // Read merges.txt
        let merges_content = std::fs::read_to_string(merges_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("Failed to read merges file: {}", e)
            ))?;

        // Parse merges (skip header line)
        let merges: Vec<String> = merges_content
            .lines()
            .skip(1) // Skip "#version: 0.2" header
            .filter(|line| !line.trim().is_empty())
            .map(|line| line.to_string())
            .collect();

        // Build tokens following BPE structure:
        // First 256 tokens should be individual bytes (0-255)
        // Then tokens built from merges
        let mut tokens: Vec<Vec<u8>> = Vec::new();
        let mut token_id_map: HashMap<u32, u32> = HashMap::new();
        let mut reverse_token_id_map: HashMap<u32, u32> = HashMap::new();
        
        // First, add all 256 individual bytes
        for byte in 0u8..=255u8 {
            tokens.push(vec![byte]);
        }
        
        // Track which byte sequences we've seen to avoid duplicates
        let mut seen_bytes: HashMap<Vec<u8>, u32> = HashMap::new();
        for (idx, token) in tokens.iter().enumerate() {
            seen_bytes.insert(token.clone(), idx as u32);
        }
        
        // Special case: Add 'ĠĠ' (two spaces) if not already present
        // This is token 256 in the Qwen vocab
        let double_space = vec![b' ', b' '];
        if !seen_bytes.contains_key(&double_space) {
            let new_idx = tokens.len() as u32;
            seen_bytes.insert(double_space.clone(), new_idx);
            tokens.push(double_space);
        }
        
        // Now process merges to build composite tokens
        // Each merge combines two existing tokens
        let mut duplicate_count = 0;
        let mut skipped_merges = Vec::new();
        for (idx, merge_line) in merges.iter().enumerate() {
            let parts: Vec<&str> = merge_line.split_whitespace().collect();
            if parts.len() != 2 {
                continue;
            }
            
            // Decode the two parts
            let left_bytes = decode_gpt2_bytes(parts[0]);
            let right_bytes = decode_gpt2_bytes(parts[1]);
            
            // Combine them
            let mut combined = left_bytes.clone();
            combined.extend(&right_bytes);
            
            // Only add if we haven't seen this byte sequence
            if !seen_bytes.contains_key(&combined) {
                let new_idx = tokens.len() as u32;
                seen_bytes.insert(combined.clone(), new_idx);
                tokens.push(combined);
            } else {
                duplicate_count += 1;
                if duplicate_count <= 10 {
                    skipped_merges.push(format!("Merge {}: {} + {}", idx, parts[0], parts[1]));
                }
            }
        }
        
        // Now map vocab entries to token indices
        for (token_str, &token_id) in vocab.iter() {
            let bytes = decode_gpt2_bytes(token_str);
            
            if let Some(&token_idx) = seen_bytes.get(&bytes) {
                token_id_map.insert(token_id, token_idx);
                reverse_token_id_map.entry(token_idx).or_insert(token_id);
            }
        }
        
        // Deduplicate tokens to satisfy BPE crate requirements
        let mut dedup_tokens: Vec<Vec<u8>> = Vec::new();
        let mut dedup_seen: HashMap<Vec<u8>, u32> = HashMap::new();
        
        for token in tokens {
            if !dedup_seen.contains_key(&token) {
                let idx = dedup_tokens.len() as u32;
                dedup_seen.insert(token.clone(), idx);
                dedup_tokens.push(token);
            }
        }
        
        // Create BPE using the rust-gems fast implementation
        // Find the proper hash factor to avoid collisions
        let hash_factor = bpe::byte_pair_encoding::find_hash_factor_for_dictionary(dedup_tokens.clone());
        
        let bpe = BytePairEncoding::from_dictionary(
            dedup_tokens,
            Some(hash_factor)
        );

        // Use default pretokenization regex for GPT-style tokenizers
        let regex_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        let pre_tokenizer_regex = Regex::new(regex_str).ok();

        // Basic special tokens for Qwen
        let mut special_tokens = HashMap::new();
        let mut special_token_ids = HashMap::new();
        
        // Common Qwen special tokens
        let special_token_strings = vec![
            "<|endoftext|>", "<|im_start|>", "<|im_end|>",
            "<|object_ref_start|>", "<|object_ref_end|>",
            "<|box_start|>", "<|box_end|>",
            "<|quad_start|>", "<|quad_end|>",
            "<|vision_start|>", "<|vision_end|>"
        ];
        
        for special_str in special_token_strings {
            if let Some(&token_id) = vocab.get(special_str) {
                special_tokens.insert(special_str.to_string(), token_id);
                special_token_ids.insert(token_id, special_str.to_string());
            }
        }

        Ok(QwenTokenizer {
            bpe,
            pre_tokenizer_regex,
            normalizer_type: Some("NFC".to_string()),
            special_tokens,
            special_token_ids,
            token_id_map,
            reverse_token_id_map,
        })
    }

    /// Encode text to token IDs
    fn encode(&self, text: &str) -> PyResult<Vec<u32>> {
        // Apply normalization if configured
        let normalized = if let Some(ref norm_type) = self.normalizer_type {
            match norm_type.as_str() {
                "NFC" => text.nfc().collect::<String>(),
                "NFD" => text.nfd().collect::<String>(),
                "NFKC" => text.nfkc().collect::<String>(),
                "NFKD" => text.nfkd().collect::<String>(),
                _ => text.to_string(),
            }
        } else {
            text.to_string()
        };

        // Handle special tokens
        for (special_text, &special_id) in &self.special_tokens {
            if normalized.contains(special_text) {
                // Split by special token and encode parts
                let mut result = Vec::new();
                let parts: Vec<&str> = normalized.split(special_text).collect();
                for (i, part) in parts.iter().enumerate() {
                    if !part.is_empty() {
                        result.extend(self.encode_regular(part)?);
                    }
                    if i < parts.len() - 1 {
                        result.push(special_id);
                    }
                }
                return Ok(result);
            }
        }

        self.encode_regular(&normalized)
    }

    /// Internal method to encode regular text
    fn encode_regular(&self, text: &str) -> PyResult<Vec<u32>> {
        if let Some(ref regex) = self.pre_tokenizer_regex {
            // Apply pre-tokenization using regex
            let mut all_tokens = Vec::new();
            
            for mat in regex.find_iter(text) {
                let piece = mat.as_str();
                // Convert to bytes for BPE encoding
                let piece_bytes = piece.as_bytes();
                // Use the fast BPE encoding from rust-gems
                let dedup_tokens = self.bpe.encode_via_backtracking(piece_bytes);
                // Map deduplicated indices back to original token IDs
                for dedup_token in dedup_tokens {
                    if let Some(&orig_id) = self.reverse_token_id_map.get(&dedup_token) {
                        all_tokens.push(orig_id);
                    } else {
                        all_tokens.push(dedup_token);  // Fallback to the dedup token if no mapping
                    }
                }
            }
            
            Ok(all_tokens)
        } else {
            // No pre-tokenization, encode the full text
            let dedup_tokens = self.bpe.encode_via_backtracking(text.as_bytes());
            // Map deduplicated indices back to original token IDs
            let mut tokens = Vec::new();
            for dedup_token in dedup_tokens {
                if let Some(&orig_id) = self.reverse_token_id_map.get(&dedup_token) {
                    tokens.push(orig_id);
                } else {
                    tokens.push(dedup_token);  // Fallback to the dedup token if no mapping
                }
            }
            Ok(tokens)
        }
    }

    /// Decode token IDs back to text
    fn decode(&self, tokens: Vec<u32>) -> PyResult<String> {
        // Map original token IDs to deduplicated indices
        let mut dedup_tokens = Vec::new();
        for token in tokens {
            if let Some(&dedup_id) = self.token_id_map.get(&token) {
                dedup_tokens.push(dedup_id);
            } else {
                dedup_tokens.push(token);  // Fallback to original if not in map
            }
        }
        
        let bytes = self.bpe.decode_tokens(&dedup_tokens);
        Ok(String::from_utf8_lossy(&bytes).to_string())
    }

    /// Encode text to token IDs and return as bytes
    fn encode_bytes(&self, py: Python, text: &str) -> PyResult<PyObject> {
        let tokens = self.encode(text)?;
        let bytes: Vec<u8> = tokens.iter().flat_map(|&t| t.to_le_bytes()).collect();
        Ok(PyBytes::new_bound(py, &bytes).into())
    }

    /// Get vocabulary size
    fn vocab_size(&self) -> usize {
        self.bpe.num_tokens() + self.special_tokens.len()
    }

    /// Count tokens without full encoding (fast!)
    fn count_tokens(&self, text: &str) -> PyResult<usize> {
        // Apply normalization
        let normalized = if let Some(ref norm_type) = self.normalizer_type {
            match norm_type.as_str() {
                "NFC" => text.nfc().collect::<String>(),
                _ => text.to_string(),
            }
        } else {
            text.to_string()
        };

        // Use the fast count method from BPE
        let mut count = 0;
        
        if let Some(ref regex) = self.pre_tokenizer_regex {
            for mat in regex.find_iter(&normalized) {
                let piece_bytes = mat.as_str().as_bytes();
                count += self.bpe.count(piece_bytes);
            }
        } else {
            count = self.bpe.count(normalized.as_bytes());
        }
        
        Ok(count)
    }
}



/// Decode GPT2-style byte representation to actual bytes
/// This handles the special encoding where spaces are Ġ and other bytes have specific mappings
fn decode_gpt2_bytes(token: &str) -> Vec<u8> {
    // GPT2 byte-level BPE uses a special encoding for bytes
    // The mapping is defined in the original GPT-2 tokenizer
    
    fn bytes_to_unicode() -> HashMap<u8, char> {
        let mut bs: Vec<u8> = vec![];
        // Add printable ASCII
        bs.extend(b'!'..=b'~');
        bs.extend(b'\xA1'..=b'\xAC');
        bs.extend(b'\xAE'..=b'\xFF');
        
        let mut cs: Vec<char> = bs.iter().map(|&b| b as char).collect();
        
        let mut n = 0u32;
        for b in 0u8..=255u8 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(char::from_u32(256 + n).unwrap());
                n += 1;
            }
        }
        
        bs.into_iter().zip(cs).collect()
    }
    
    // Create the reverse mapping
    let unicode_to_bytes: HashMap<char, u8> = bytes_to_unicode()
        .into_iter()
        .map(|(b, c)| (c, b))
        .collect();
    
    let mut bytes = Vec::new();
    for ch in token.chars() {
        if let Some(&byte) = unicode_to_bytes.get(&ch) {
            bytes.push(byte);
        } else {
            // If not in mapping, it's probably a multi-byte UTF-8 character
            let ch_str = ch.to_string();
            bytes.extend_from_slice(ch_str.as_bytes());
        }
    }
    
    bytes
}

/// Python module definition
#[pymodule]
fn bpe_qwen(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<QwenTokenizer>()?;
    Ok(())
}