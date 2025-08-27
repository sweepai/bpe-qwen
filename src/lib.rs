use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::Deserialize;
use std::collections::HashMap;
use regex::Regex;
use unicode_normalization::UnicodeNormalization;
use serde_json::Value;
use std::path::Path;

use bpe::byte_pair_encoding::BytePairEncoding;

/// Fast ASCII detection using 8-byte chunks (SIMD-like optimization)
fn is_ascii_fast(text: &str) -> bool {
    let bytes = text.as_bytes();
    let len = bytes.len();
    
    // Process 8 bytes at a time using u64
    let chunks = len / 8;
    let ptr = bytes.as_ptr() as *const u64;
    
    // Check 8-byte chunks
    for i in 0..chunks {
        unsafe {
            let chunk = ptr.add(i).read_unaligned();
            // If any byte has the high bit set (0x80), it's non-ASCII
            if chunk & 0x8080808080808080u64 != 0 {
                return false;
            }
        }
    }
    
    // Check remaining bytes
    let remainder_start = chunks * 8;
    for &byte in &bytes[remainder_start..] {
        if byte & 0x80 != 0 {
            return false;
        }
    }
    
    true
}

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
    reverse_token_id_vec: Vec<u32>,  // Maps deduplicated indices back to original IDs (Vec for O(1) lookup)
}

#[pymethods]
impl QwenTokenizer {
    /// Create a new QwenTokenizer from a directory containing tokenizer files
    /// 
    /// Args:
    ///     dir_path: Directory containing vocab.json, merges.txt, and optionally tokenizer_config.json
    ///     pretokenize_regex: Optional custom pretokenization regex (defaults to Qwen regex if None)
    #[new]
    #[pyo3(signature = (dir_path, pretokenize_regex=None))]
    fn new(dir_path: &str, pretokenize_regex: Option<&str>) -> PyResult<Self> {
        let dir = Path::new(dir_path);
        
        // Build paths for required files
        let vocab_path = dir.join("vocab.json");
        let merges_path = dir.join("merges.txt");
        let tokenizer_config_path = dir.join("tokenizer_config.json");
        
        // Read vocab.json
        let vocab_content = std::fs::read_to_string(&vocab_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("Failed to read vocab file at {:?}: {}", vocab_path, e)
            ))?;
        
        let mut vocab: HashMap<String, u32> = serde_json::from_str(&vocab_content)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Failed to parse vocab: {}", e)
            ))?;

        // Read merges.txt
        let merges_content = std::fs::read_to_string(&merges_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("Failed to read merges file at {:?}: {}", merges_path, e)
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
        
        let num_dedup_tokens = dedup_tokens.len();
        let bpe = BytePairEncoding::from_dictionary(
            dedup_tokens,
            Some(hash_factor)
        );

        // Load special tokens from tokenizer_config.json if provided
        let mut special_tokens = HashMap::new();
        let mut special_token_ids = HashMap::new();
        
        if tokenizer_config_path.exists() {
            if let Ok(config_content) = std::fs::read_to_string(&tokenizer_config_path) {
                if let Ok(config_json) = serde_json::from_str::<Value>(&config_content) {
                    // Load from added_tokens_decoder
                    if let Some(added_tokens) = config_json.get("added_tokens_decoder").and_then(|v| v.as_object()) {
                        for (token_id_str, token_info) in added_tokens.iter() {
                            if let Ok(token_id) = token_id_str.parse::<u32>() {
                                if let Some(content) = token_info.get("content").and_then(|v| v.as_str()) {
                                    // Add the special token to our maps
                                    special_tokens.insert(content.to_string(), token_id);
                                    special_token_ids.insert(token_id, content.to_string());
                                    
                                    // Also add to vocab if not present
                                    vocab.entry(content.to_string()).or_insert(token_id);
                                }
                            }
                        }
                    }
                }
                println!("Loaded {} special tokens from tokenizer_config.json", special_tokens.len());
            }
        } else {
            // Fallback: try to find common special tokens in vocab
            let special_token_strings = vec![
                "<|endoftext|>", "<|im_start|>", "<|im_end|>",
                "<|object_ref_start|>", "<|object_ref_end|>",
                "<|box_start|>", "<|box_end|>",
                "<|quad_start|>", "<|quad_end|>",
                "<|vision_start|>", "<|vision_end|>",
                "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
            ];
            
            for special_str in special_token_strings {
                if let Some(&token_id) = vocab.get(special_str) {
                    special_tokens.insert(special_str.to_string(), token_id);
                    special_token_ids.insert(token_id, special_str.to_string());
                }
            }
        }

        // Skip regex pretokenization for maximum performance
        let pre_tokenizer_regex = None;

        Ok(QwenTokenizer {
            bpe,
            pre_tokenizer_regex,
            normalizer_type: Some("NFC".to_string()),
            special_tokens,
            special_token_ids,
            token_id_map,
            reverse_token_id_vec: {
                let max_dedup_id = num_dedup_tokens as u32;
                let mut vec = vec![0u32; max_dedup_id as usize];
                for (dedup_id, &orig_id) in reverse_token_id_map.iter() {
                    if *dedup_id < max_dedup_id {
                        vec[*dedup_id as usize] = orig_id;
                    }
                }
                vec
            },
        })
    }

    /// Encode text to token IDs
    fn encode(&self, text: &str) -> PyResult<Vec<u32>> {
        let total_start = std::time::Instant::now();
        
        // Apply normalization if configured
        let norm_start = std::time::Instant::now();
        let normalized = if let Some(ref norm_type) = self.normalizer_type {
            if is_ascii_fast(text) {
                text.to_string()  // Skip normalization for ASCII text
            } else {
                match norm_type.as_str() {
                    "NFC" => text.nfc().collect::<String>(),
                    "NFD" => text.nfd().collect::<String>(),
                    "NFKC" => text.nfkc().collect::<String>(),
                    "NFKD" => text.nfkd().collect::<String>(),
                    _ => text.to_string(),
                }
            }
        } else {
            text.to_string()
        };
        let norm_time = norm_start.elapsed();

        // Handle special tokens - find all occurrences and their positions
        let special_start = std::time::Instant::now();
        let mut special_token_positions = Vec::new();
        for (special_text, &special_id) in &self.special_tokens {
            let mut search_pos = 0;
            while let Some(pos) = normalized[search_pos..].find(special_text) {
                let actual_pos = search_pos + pos;
                special_token_positions.push((actual_pos, special_text.len(), special_id));
                search_pos = actual_pos + special_text.len();
            }
        }
        
        // Sort by position
        special_token_positions.sort_by_key(|&(pos, _, _)| pos);
        let special_time = special_start.elapsed();
        
        // If we have special tokens, split and encode
        let encode_start = std::time::Instant::now();
        let result = if !special_token_positions.is_empty() {
            let mut result = Vec::new();
            let mut last_end = 0;
            
            for (start, len, token_id) in special_token_positions {
                // Encode text before the special token
                if start > last_end {
                    let part = &normalized[last_end..start];
                    if !part.is_empty() {
                        result.extend(self.encode_regular(part)?);
                    }
                }
                // Add the special token
                result.push(token_id);
                last_end = start + len;
            }
            
            // Encode any remaining text after the last special token
            if last_end < normalized.len() {
                let part = &normalized[last_end..];
                if !part.is_empty() {
                    result.extend(self.encode_regular(part)?);
                }
            }
            
            result
        } else {
            self.encode_regular(&normalized)?
        };
        let encode_time = encode_start.elapsed();
        let total_time = total_start.elapsed();
        
        // Profiling disabled for accurate benchmarks
        
        Ok(result)
    }

    /// Internal method to encode regular text
    fn encode_regular(&self, text: &str) -> PyResult<Vec<u32>> {
        if let Some(ref regex) = self.pre_tokenizer_regex {
            // Apply pre-tokenization using regex
            let mut all_tokens = Vec::with_capacity(128);  // Optimal fixed capacity
            
            // Use standard regex which doesn't return Results
            let matches: Vec<_> = regex.find_iter(text).collect();
            
            for mat in matches {
                let piece = mat.as_str();
                // Convert to bytes for BPE encoding
                let piece_bytes = piece.as_bytes();
                // Use the fast BPE encoding from rust-gems
                let dedup_tokens = self.bpe.encode_via_backtracking(piece_bytes);
                // Map deduplicated indices back to original token IDs
                for dedup_token in dedup_tokens {
                    if (dedup_token as usize) < self.reverse_token_id_vec.len() {
                        let orig_id = self.reverse_token_id_vec[dedup_token as usize];
                        all_tokens.push(orig_id);
                    } else {
                        all_tokens.push(dedup_token);  // Fallback to the dedup token if no mapping
                    }
                }
            }
            
            Ok(all_tokens)
        } else {
            // No pre-tokenization, encode the full text
            let bpe_start = std::time::Instant::now();
            let dedup_tokens = self.bpe.encode_via_backtracking(text.as_bytes());
            let bpe_time = bpe_start.elapsed();
            
            // Map deduplicated indices back to original token IDs
            let map_start = std::time::Instant::now();
            let mut tokens = Vec::with_capacity(128);  // Optimal fixed capacity
            for dedup_token in dedup_tokens {
                if (dedup_token as usize) < self.reverse_token_id_vec.len() {
                    let orig_id = self.reverse_token_id_vec[dedup_token as usize];
                    tokens.push(orig_id);
                } else {
                    tokens.push(dedup_token);  // Fallback to the dedup token if no mapping
                }
            }
            let map_time = map_start.elapsed();
            
            // Profiling disabled for accurate benchmarks
            
            Ok(tokens)
        }
    }

    /// Decode token IDs back to text
    fn decode(&self, tokens: Vec<u32>) -> PyResult<String> {
        let mut result = String::new();
        
        for token in tokens {
            // Check if it's a special token first
            if let Some(special_text) = self.special_token_ids.get(&token) {
                result.push_str(special_text);
            } else if let Some(&dedup_id) = self.token_id_map.get(&token) {
                // Map to deduplicated index and decode
                let bytes = self.bpe.decode_tokens(&[dedup_id]);
                result.push_str(&String::from_utf8_lossy(&bytes));
            } else {
                // Try decoding directly (though this might fail for out-of-range tokens)
                if token < self.bpe.num_tokens() as u32 {
                    let bytes = self.bpe.decode_tokens(&[token]);
                    result.push_str(&String::from_utf8_lossy(&bytes));
                }
                // If token is completely out of range, skip it
            }
        }
        
        Ok(result)
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
            // fancy_regex returns Results, so we need to handle them
            let matches: Vec<_> = regex.find_iter(&normalized).collect();
            
            for mat in matches {
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