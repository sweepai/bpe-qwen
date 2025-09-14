use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::HashMap;
use unicode_normalization::UnicodeNormalization;
use serde_json::Value;
use std::path::Path;

use bpe::byte_pair_encoding::BytePairEncoding;

mod pretokenization;
mod pretokenization_indices;
use std::cell::RefCell;
use std::borrow::Cow;

/// Memory pool for reusing Vec<u32> allocations
struct VectorPool {
    available: RefCell<Vec<Vec<u32>>>,
}

impl VectorPool {
    fn new() -> Self {
        Self {
            available: RefCell::new(Vec::new()),
        }
    }
    
    fn get_buffer(&self, min_capacity: usize) -> Vec<u32> {
        let mut available = self.available.borrow_mut();
        
        // Find a buffer with sufficient capacity
        for i in 0..available.len() {
            if available[i].capacity() >= min_capacity {
                let mut buf = available.swap_remove(i);
                buf.clear();
                return buf;
            }
        }
        
        // No suitable buffer found, create a new one
        Vec::with_capacity(min_capacity.max(128))
    }
    

}

/// Fast ASCII detection using true SIMD intrinsics
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
fn is_ascii_fast(text: &str) -> bool {
    use std::arch::x86_64::*;
    
    let bytes = text.as_bytes();
    let len = bytes.len();
    
    // Process 16 bytes at a time with SSE2
    let chunks = len / 16;
    let ptr = bytes.as_ptr();
    
    unsafe {
        let mask = _mm_set1_epi8(0x80u8 as i8);
        
        for i in 0..chunks {
            let data = _mm_loadu_si128(ptr.add(i * 16) as *const __m128i);
            let result = _mm_and_si128(data, mask);
            if _mm_movemask_epi8(result) != 0 {
                return false;
            }
        }
    }
    
    // Check remaining bytes
    let remainder_start = chunks * 16;
    for &byte in &bytes[remainder_start..] {
        if byte & 0x80 != 0 {
            return false;
        }
    }
    
    true
}

/// Fast ASCII detection using NEON SIMD for ARM (Apple Silicon)
#[cfg(target_arch = "aarch64")]
fn is_ascii_fast(text: &str) -> bool {
    use std::arch::aarch64::*;
    
    let bytes = text.as_bytes();
    let len = bytes.len();
    
    // Process 16 bytes at a time with NEON
    let chunks = len / 16;
    let ptr = bytes.as_ptr();
    
    unsafe {
        for i in 0..chunks {
            let data = vld1q_u8(ptr.add(i * 16));
            let max_val = vmaxvq_u8(data);
            if max_val >= 0x80 {
                return false;
            }
        }
    }
    
    // Check remaining bytes
    let remainder_start = chunks * 16;
    for &byte in &bytes[remainder_start..] {
        if byte & 0x80 != 0 {
            return false;
        }
    }
    
    true
}

/// Fallback ASCII detection for other architectures
#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    target_arch = "aarch64"
)))]
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



/// Helper function for parallel encoding without vector pool using indices
fn encode_text_parallel(
    text: &str,
    bpe: &BytePairEncoding,
    normalizer_type: &Option<String>,
    special_tokens: &HashMap<String, u32>,
    reverse_token_id_vec: &Vec<u32>,
) -> Result<Vec<u32>, String> {
    // Check if text is empty
    if text.is_empty() {
        return Ok(Vec::new());
    }
    
    // Create a fresh buffer for this thread
    let mut all_tokens = Vec::with_capacity(128);

    // Use the indices-based pretokenization for better performance
    let end_indices = crate::pretokenization_indices::pretokenize_fast_indices(text);

    let mut start = 0;
    for &end in &end_indices {
        let piece = &text[start..end];
        start = end;

        if piece.is_empty() {
            continue;
        }

        // Check for special tokens first
        if let Some(&token_id) = special_tokens.get(piece) {
            all_tokens.push(token_id);
            continue;
        }

        // Normalization: NFC normalize only when needed (non-ASCII with NFC normalizer)
        let normalized = if let Some(ref nt) = normalizer_type {
            if nt == "NFC" && !is_ascii_fast(piece) {
                // Use Cow to avoid allocation when possible
                Cow::from(piece.nfc().collect::<String>())
            } else {
                Cow::from(piece)
            }
        } else {
            Cow::from(piece)
        };

        // BPE tokenization
        let dedup_tokens = bpe.encode_via_backtracking(normalized.as_bytes());

        // Map from deduplicated indices to original token IDs
        for dedup_id in dedup_tokens {
            let original_id = *reverse_token_id_vec
                .get(dedup_id as usize)
                .ok_or_else(|| format!("Invalid dedup ID: {}", dedup_id))?;
            all_tokens.push(original_id);
        }
    }
    
    Ok(all_tokens)
}

/// QwenTokenizer - A fast BPE tokenizer for Qwen models
#[pyclass]
struct QwenTokenizer {
    bpe: BytePairEncoding,
    normalizer_type: Option<String>,
    special_tokens: HashMap<String, u32>,
    special_token_ids: HashMap<u32, String>,
    token_id_map: HashMap<u32, u32>,  // Maps original token IDs to deduplicated indices
    reverse_token_id_vec: Vec<u32>,  // Maps deduplicated indices back to original IDs (Vec for O(1) lookup)
    vector_pool: VectorPool,  // Pool for reusing Vec<u32> allocations
}

#[pymethods]
impl QwenTokenizer {
    /// Create a new QwenTokenizer from a directory containing tokenizer files
    /// 
    /// Args:
    ///     dir_path: Directory containing vocab.json, merges.txt, and optionally tokenizer_config.json
    #[new]
    fn new(dir_path: &str) -> PyResult<Self> {
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

        // Pre-tokenization will use the globally precompiled regex

        Ok(QwenTokenizer {
            bpe,
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
            vector_pool: VectorPool::new(),
        })
    }

    /// Encode text to token IDs
    fn encode(&self, text: &str) -> PyResult<Vec<u32>> {
        let total_start = std::time::Instant::now();
        
        // Apply normalization if configured - use Cow to avoid allocation when not normalizing
        let norm_start = std::time::Instant::now();
        let normalized: Cow<str> = if let Some(ref norm_type) = self.normalizer_type {
            if is_ascii_fast(text) {
                Cow::Borrowed(text)  // Zero-copy for ASCII text
            } else {
                match norm_type.as_str() {
                    "NFC" => Cow::Owned(text.nfc().collect::<String>()),
                    "NFD" => Cow::Owned(text.nfd().collect::<String>()),
                    "NFKC" => Cow::Owned(text.nfkc().collect::<String>()),
                    "NFKD" => Cow::Owned(text.nfkd().collect::<String>()),
                    _ => Cow::Borrowed(text),
                }
            }
        } else {
            Cow::Borrowed(text)  // Zero-copy when no normalization
        };
        let _norm_time = norm_start.elapsed();

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
        let _special_time = special_start.elapsed();
        
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
        let _encode_time = encode_start.elapsed();
        let _total_time = total_start.elapsed();
        
        // Profiling disabled for accurate benchmarks
        
        Ok(result)
    }

    /// Internal method to encode regular text using indices-based pretokenization
    fn encode_regular(&self, text: &str) -> PyResult<Vec<u32>> {
        // Apply pre-tokenization using indices for better performance
        let mut all_tokens = self.vector_pool.get_buffer(128);  // Get from pool

        // Use the indices-based pretokenization for better performance
        let end_indices = crate::pretokenization_indices::pretokenize_fast_indices(text);

        // Pass 1: Collect all dedup tokens from BPE encoding
        let mut all_dedup_tokens = Vec::new();
        let mut start = 0;
        for &end in &end_indices {
            let piece = &text[start..end];
            start = end;

            if piece.is_empty() {
                continue;
            }
            // Convert to bytes for BPE encoding
            let piece_bytes = piece.as_bytes();
            // Use the fast BPE encoding from rust-gems
            let dedup_tokens = self.bpe.encode_via_backtracking(piece_bytes);
            all_dedup_tokens.extend(dedup_tokens);
        }

        // Pass 2: Batch convert dedup tokens to original IDs
        for dedup_token in all_dedup_tokens {
            if (dedup_token as usize) < self.reverse_token_id_vec.len() {
                let orig_id = self.reverse_token_id_vec[dedup_token as usize];
                all_tokens.push(orig_id);
            } else {
                all_tokens.push(dedup_token);  // Fallback to the dedup token if no mapping
            }
        }

        Ok(all_tokens)
    }
    
    /// Thread-safe encode for parallel processing (doesn't use vector pool)
    fn encode_for_parallel(&self, text: &str) -> PyResult<Vec<u32>> {
        // Apply normalization if configured - use Cow to avoid allocation when not normalizing
        let normalized: Cow<str> = if let Some(ref norm_type) = self.normalizer_type {
            if is_ascii_fast(text) {
                Cow::Borrowed(text)  // Zero-copy for ASCII text
            } else {
                match norm_type.as_str() {
                    "NFC" => Cow::Owned(text.nfc().collect::<String>()),
                    "NFD" => Cow::Owned(text.nfd().collect::<String>()),
                    "NFKC" => Cow::Owned(text.nfkc().collect::<String>()),
                    "NFKD" => Cow::Owned(text.nfkd().collect::<String>()),
                    _ => Cow::Borrowed(text),
                }
            }
        } else {
            Cow::Borrowed(text)  // Zero-copy when no normalization
        };

        // Handle special tokens
        let mut special_token_positions = Vec::new();
        for (special_text, &special_id) in &self.special_tokens {
            let mut search_pos = 0;
            while let Some(pos) = normalized[search_pos..].find(special_text) {
                let actual_pos = search_pos + pos;
                special_token_positions.push((actual_pos, special_text.len(), special_id));
                search_pos = actual_pos + special_text.len();
            }
        }
        special_token_positions.sort_by_key(|&(pos, _, _)| pos);
        
        // Process text
        if special_token_positions.is_empty() {
            // No special tokens, encode the full text
            self.encode_regular(&normalized)
        } else {
            // Split text around special tokens and encode
            let mut all_tokens = Vec::with_capacity(128);  // Fixed capacity
            let mut last_pos = 0;
            
            for (pos, len, special_id) in special_token_positions {
                // Encode text before special token
                if pos > last_pos {
                    let text_segment = &normalized[last_pos..pos];
                    if !text_segment.is_empty() {
                        let tokens = self.encode_regular(text_segment)?;
                        all_tokens.extend(tokens);
                    }
                }
                // Add special token
                all_tokens.push(special_id);
                last_pos = pos + len;
            }
            
            // Encode remaining text after last special token
            if last_pos < normalized.len() {
                let text_segment = &normalized[last_pos..];
                if !text_segment.is_empty() {
                    let tokens = self.encode_regular(text_segment)?;
                    all_tokens.extend(tokens);
                }
            }
            
            Ok(all_tokens)
        }
    }
    
    /// Batch encode multiple texts in parallel using Rayon
    #[pyo3(signature = (texts, num_workers=None))]
    fn encode_batch_parallel(&self, texts: Vec<String>, num_workers: Option<usize>) -> PyResult<Vec<Vec<u32>>> {
        use rayon::prelude::*;
        use std::sync::Arc;
        
        // Share read-only data across threads using Arc
        let bpe = Arc::new(&self.bpe);
        let normalizer_type = Arc::new(&self.normalizer_type);
        let special_tokens = Arc::new(&self.special_tokens);
        let reverse_token_id_vec = Arc::new(&self.reverse_token_id_vec);
        
        // Use a scoped thread pool if num_workers is specified, otherwise use global pool
        if let Some(workers) = num_workers {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to create thread pool: {}", e)))?;
                
            let results: Vec<Vec<u32>> = pool.install(|| {
                texts
                    .par_iter()
                    .map(|text| {
                        encode_text_parallel(
                            text,
                            &**bpe,
                            &**normalizer_type,
                            &**special_tokens,
                            &**reverse_token_id_vec,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()
            }).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
            
            Ok(results)
        } else {
            // Use global thread pool
            let results: Vec<Vec<u32>> = texts
                .par_iter()
                .map(|text| {
                    encode_text_parallel(
                        text,
                        &**bpe,
                        &**normalizer_type,
                        &**special_tokens,
                        &**reverse_token_id_vec,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
            
            Ok(results)
        }
    }

    /// Decode token IDs back to text
    fn decode(&self, tokens: Vec<u32>) -> PyResult<String> {
        let mut result = String::new();
        let mut all_bytes = Vec::new();

        for token in tokens {
            // Check if it's a special token first
            if let Some(special_text) = self.special_token_ids.get(&token) {
                // For special tokens, we need to flush any accumulated bytes first
                if !all_bytes.is_empty() {
                    // Convert accumulated bytes to string and append to result
                    result.push_str(&String::from_utf8_lossy(&all_bytes));
                    all_bytes.clear();
                }
                // Special tokens are already valid UTF-8 strings, append directly
                result.push_str(special_text);
            } else if let Some(&dedup_id) = self.token_id_map.get(&token) {
                // Map to deduplicated index and decode
                let bytes = self.bpe.decode_tokens(&[dedup_id]);
                all_bytes.extend_from_slice(&bytes);
            } else {
                // Try decoding directly (though this might fail for out-of-range tokens)
                if token < self.bpe.num_tokens() as u32 {
                    let bytes = self.bpe.decode_tokens(&[token]);
                    all_bytes.extend_from_slice(&bytes);
                }
                // If token is completely out of range, skip it
            }
        }

        // Convert any remaining bytes to string at the end
        // This preserves multi-byte UTF-8 sequences that span multiple tokens
        if !all_bytes.is_empty() {
            result.push_str(&String::from_utf8_lossy(&all_bytes));
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

    /// Count tokens without full encoding (fast!) using indices-based pretokenization
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

        // Use the fast count method from BPE with indices-based pretokenization
        let mut count = 0;
        let end_indices = crate::pretokenization_indices::pretokenize_fast_indices(&normalized);

        let mut start = 0;
        for &end in &end_indices {
            let piece = &normalized[start..end];
            start = end;

            if !piece.is_empty() {
                let piece_bytes = piece.as_bytes();
                count += self.bpe.count(piece_bytes);
            }
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

/// Expose the slow pretokenization (using fancy-regex with lookahead)
#[pyfunction]
fn pretokenize_slow(text: &str) -> PyResult<Vec<String>> {
    Ok(pretokenization::pretokenize_slow(text))
}

/// Expose the fast pretokenization (using standard regex with correction pass)
#[pyfunction]
fn pretokenize_fast(text: &str) -> PyResult<Vec<String>> {
    Ok(pretokenization::pretokenize_fast(text))
}

/// Expose the indices-based pretokenization (returns end positions only)
#[pyfunction]
fn pretokenize_fast_indices(text: &str) -> PyResult<Vec<usize>> {
    Ok(pretokenization_indices::pretokenize_fast_indices(text))
}

/// Convert end indices to strings for testing
#[pyfunction]
fn indices_to_strings(text: &str, end_indices: Vec<usize>) -> PyResult<Vec<String>> {
    Ok(pretokenization_indices::indices_to_strings(text, &end_indices))
}

/// Python module definition
#[pymodule]
fn bpe_qwen(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<QwenTokenizer>()?;
    m.add_function(wrap_pyfunction!(pretokenize_slow, m)?)?;
    m.add_function(wrap_pyfunction!(pretokenize_fast, m)?)?;
    m.add_function(wrap_pyfunction!(pretokenize_fast_indices, m)?)?;
    m.add_function(wrap_pyfunction!(indices_to_strings, m)?)?;
    Ok(())
}