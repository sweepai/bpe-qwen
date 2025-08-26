use std::collections::HashMap;
use regex::Regex;

pub struct SimpleQwenTokenizer {
    pub vocab: HashMap<String, u32>,
    pub id_to_token: HashMap<u32, String>,
    pub pre_tokenizer_regex: Option<Regex>,
    pub special_tokens: HashMap<String, u32>,
}

impl SimpleQwenTokenizer {
    pub fn new(
        vocab: HashMap<String, u32>,
        pre_tokenizer_regex: Option<Regex>,
        special_tokens: HashMap<String, u32>,
    ) -> Self {
        let mut id_to_token = HashMap::new();
        for (token, &id) in vocab.iter() {
            id_to_token.insert(id, token.clone());
        }
        
        SimpleQwenTokenizer {
            vocab,
            id_to_token,
            pre_tokenizer_regex,
            special_tokens,
        }
    }
    
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // Simple greedy tokenization
        // This is not the full BPE algorithm but a simplified version
        let mut tokens = Vec::new();
        
        // Check for special tokens first
        for (special_text, &special_id) in &self.special_tokens {
            if text.contains(special_text) {
                // Handle special tokens by splitting
                let parts: Vec<&str> = text.split(special_text).collect();
                for (i, part) in parts.iter().enumerate() {
                    if !part.is_empty() {
                        tokens.extend(self.encode_regular(part));
                    }
                    if i < parts.len() - 1 {
                        tokens.push(special_id);
                    }
                }
                return tokens;
            }
        }
        
        self.encode_regular(text)
    }
    
    fn encode_regular(&self, text: &str) -> Vec<u32> {
        if let Some(ref regex) = self.pre_tokenizer_regex {
            let mut all_tokens = Vec::new();
            
            for mat in regex.find_iter(text) {
                let piece = mat.as_str();
                // Use greedy matching for now
                all_tokens.extend(self.greedy_tokenize(piece));
            }
            
            all_tokens
        } else {
            self.greedy_tokenize(text)
        }
    }
    
    fn greedy_tokenize(&self, text: &str) -> Vec<u32> {
        // Simple greedy tokenization - find longest matching tokens
        let mut tokens = Vec::new();
        let mut pos = 0;
        let bytes = text.as_bytes();
        
        while pos < bytes.len() {
            let mut longest_match = None;
            let mut longest_len = 0;
            
            // Try to find the longest matching token from current position
            for len in (1..=bytes.len() - pos).rev() {
                let substr = std::str::from_utf8(&bytes[pos..pos + len]);
                if let Ok(s) = substr {
                    if let Some(&token_id) = self.vocab.get(s) {
                        longest_match = Some(token_id);
                        longest_len = len;
                        break;
                    }
                }
            }
            
            if let Some(token_id) = longest_match {
                tokens.push(token_id);
                pos += longest_len;
            } else {
                // Fall back to byte-level tokens if no match found
                // Convert byte to the GPT2-style token representation
                let byte_char = encode_byte_char(bytes[pos]);
                let byte_str = byte_char.to_string();
                if let Some(&token_id) = self.vocab.get(&byte_str) {
                    tokens.push(token_id);
                }
                pos += 1;
            }
        }
        
        tokens
    }
    
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut result = String::new();
        
        for &token_id in tokens {
            if let Some(token_str) = self.id_to_token.get(&token_id) {
                // Decode the token bytes
                let bytes = decode_token_bytes(token_str);
                if let Ok(s) = std::str::from_utf8(&bytes) {
                    result.push_str(s);
                } else {
                    result.push_str(&String::from_utf8_lossy(&bytes));
                }
            }
        }
        
        result
    }
    
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

/// Encode a byte to GPT2 byte-level representation
fn encode_byte_char(byte: u8) -> char {
    match byte {
        // Direct ASCII printable range  
        b' ' => ' ',
        b'!' => '!',
        b'"' => '"',
        b'#' => '#',
        b'$' => '$',
        b'%' => '%',
        b'&' => '&',
        b'\'' => '\'',
        b'(' => '(',
        b')' => ')',
        b'*' => '*',
        b'+' => '+',
        b',' => ',',
        b'-' => '-',
        b'.' => '.',
        b'/' => '/',
        b'0'..=b'9' => byte as char,
        b':' => ':',
        b';' => ';',
        b'<' => '<',
        b'=' => '=',
        b'>' => '>',
        b'?' => '?',
        b'@' => '@',
        b'A'..=b'Z' => byte as char,
        b'[' => '[',
        b'\\' => '\\',
        b']' => ']',
        b'^' => '^',
        b'_' => '_',
        b'`' => '`',
        b'a'..=b'z' => byte as char,
        b'{' => '{',
        b'|' => '|',
        b'}' => '}',
        b'~' => '~',
        // Special mappings for control characters and non-printable bytes
        0..=31 => char::from_u32(256 + byte as u32).unwrap(),
        127 => '\u{0120}', // 288
        128..=138 => char::from_u32(161 + (byte - 127) as u32).unwrap(),
        139..=255 => char::from_u32(174 + (byte - 138) as u32).unwrap(),
        _ => char::from_u32(byte as u32).unwrap(),
    }
}

/// Decode token bytes from GPT2-style byte-level representation  
pub fn decode_token_bytes(token: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    let chars: Vec<char> = token.chars().collect();
    let mut i = 0;
    
    while i < chars.len() {
        let ch = chars[i];
        
        // Check if it's a byte-level representation
        if let Some(byte) = decode_byte_char(ch) {
            bytes.push(byte);
        } else {
            // Regular UTF-8 character
            let ch_bytes = ch.to_string().into_bytes();
            bytes.extend(ch_bytes);
        }
        i += 1;
    }
    
    bytes
}

/// Decode a single character from GPT2 byte-level representation
fn decode_byte_char(ch: char) -> Option<u8> {
    // GPT2 byte-level BPE mapping
    match ch as u32 {
        // Direct ASCII printable range
        c if c >= 33 && c <= 126 => Some(c as u8),
        c if c >= 161 && c <= 172 => Some((c - 161 + 127) as u8),
        c if c >= 174 && c <= 255 => Some((c - 174 + 138) as u8),
        // Space
        32 => Some(32),
        // Special mappings for control characters and non-printable bytes
        256 => Some(0),
        257 => Some(1),
        258 => Some(2),
        259 => Some(3),
        260 => Some(4),
        261 => Some(5),
        262 => Some(6),
        263 => Some(7),
        264 => Some(8),
        265 => Some(9),
        266 => Some(10),
        267 => Some(11),
        268 => Some(12),
        269 => Some(13),
        270 => Some(14),
        271 => Some(15),
        272 => Some(16),
        273 => Some(17),
        274 => Some(18),
        275 => Some(19),
        276 => Some(20),
        277 => Some(21),
        278 => Some(22),
        279 => Some(23),
        280 => Some(24),
        281 => Some(25),
        282 => Some(26),
        283 => Some(27),
        284 => Some(28),
        285 => Some(29),
        286 => Some(30),
        287 => Some(31),
        288 => Some(127),
        _ => None,
    }
}