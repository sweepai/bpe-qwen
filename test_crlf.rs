// Test script to debug CRLF handling in Rust
use regex::Regex;

fn main() {
    let pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+";
    let regex = Regex::new(pattern).unwrap();

    let text = "line3\r\nline4";
    println!("Testing text: {:?}", text);
    println!();

    println!("Raw regex matches:");
    for mat in regex.find_iter(text) {
        println!("  {:?} at {}-{}", mat.as_str(), mat.start(), mat.end());
    }

    println!();
    println!("Testing whitespace check:");
    let test_str = "\r\n";
    let is_whitespace = test_str.chars().all(|c| c.is_whitespace());
    let contains_r = test_str.contains('\r');
    let contains_n = test_str.contains('\n');

    println!("  {:?} is_whitespace: {}", test_str, is_whitespace);
    println!("  {:?} contains \\r: {}", test_str, contains_r);
    println!("  {:?} contains \\n: {}", test_str, contains_n);

    let should_apply_correction = is_whitespace && !contains_r && !contains_n;
    println!("  Should apply correction: {}", should_apply_correction);
}