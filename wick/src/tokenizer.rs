use std::collections::HashMap;

use anyhow::{Context, Result};
use regex::Regex;

use crate::gguf::{GgufFile, GgufValue};

/// A segment of text split at special token boundaries.
enum Segment<'a> {
    Text(&'a str),
    Special(u32),
}

/// A minimal byte-level BPE tokenizer.
///
/// Loads vocabulary and merges from GGUF metadata. Implements the same
/// byte-level BPE algorithm used by LLaMA, LFM2, GPT-NeoX, etc.
pub struct BpeTokenizer {
    /// Token ID → token string (may contain raw bytes as escaped sequences).
    vocab: Vec<Vec<u8>>,
    /// Token string → token ID.
    token_to_id: HashMap<Vec<u8>, u32>,
    /// Merge pairs in priority order (highest priority = lowest index).
    /// Maps (token_a, token_b) → priority rank.
    merge_ranks: HashMap<(Vec<u8>, Vec<u8>), usize>,
    /// Special token name → token ID.
    special_tokens: HashMap<String, u32>,
    /// BOS token ID.
    bos_id: Option<u32>,
    /// EOS token ID.
    eos_id: Option<u32>,
    /// Chat template (Jinja2 format) if present.
    chat_template: Option<String>,
    /// Pre-compiled pretokenizer regex.
    pretokenize_re: Regex,
    /// GPT-2 byte→unicode mapping (computed once, used in encode).
    byte_to_unicode: [char; 256],
    /// GPT-2 unicode→byte mapping (computed once, used in decode).
    unicode_to_byte: HashMap<char, u8>,
}

impl BpeTokenizer {
    /// Load a tokenizer from GGUF metadata.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        // Extract vocabulary tokens
        let tokens = gguf
            .get_string_array("tokenizer.ggml.tokens")
            .context("missing tokenizer.ggml.tokens")?;

        let vocab_size = tokens.len();

        // Build vocab and reverse mapping
        let mut vocab: Vec<Vec<u8>> = Vec::with_capacity(vocab_size);
        let mut token_to_id: HashMap<Vec<u8>, u32> = HashMap::with_capacity(vocab_size);

        for (id, token_str) in tokens.iter().enumerate() {
            let token_bytes = unescape_token(token_str);
            token_to_id.insert(token_bytes.clone(), id as u32);
            vocab.push(token_bytes);
        }

        // Extract merge rules
        let mut merge_ranks: HashMap<(Vec<u8>, Vec<u8>), usize> = HashMap::new();

        if let Some(merges) = gguf.get_string_array("tokenizer.ggml.merges") {
            for (rank, merge_str) in merges.iter().enumerate() {
                // Each merge is "token_a token_b" separated by a space
                if let Some((a, b)) = merge_str.split_once(' ') {
                    let a_bytes = unescape_token(a);
                    let b_bytes = unescape_token(b);
                    merge_ranks.insert((a_bytes, b_bytes), rank);
                }
            }
        }

        // Extract special tokens
        let mut special_tokens = HashMap::new();

        // Check for token type array to identify special tokens
        if let Some(GgufValue::Array(types)) = gguf.metadata.get("tokenizer.ggml.token_type") {
            for (id, type_val) in types.iter().enumerate() {
                if let GgufValue::I32(t) = type_val {
                    // Type 3 = control token, Type 4 = user-defined special
                    if (*t == 3 || *t == 4) && id < vocab_size {
                        let token_str = &tokens[id];
                        special_tokens.insert(token_str.to_string(), id as u32);
                    }
                }
            }
        }

        // Extract BOS/EOS token IDs
        let bos_id = gguf.get_u32("tokenizer.ggml.bos_token_id");
        let eos_id = gguf.get_u32("tokenizer.ggml.eos_token_id");

        // Extract chat template
        let chat_template = gguf.get_str("tokenizer.chat_template").map(String::from);

        // Select pretokenizer based on model type
        let pre_type = gguf.get_str("tokenizer.ggml.pre").unwrap_or("gpt2");
        let pretokenize_re = build_pretokenize_regex(pre_type);
        let byte_to_unicode = build_byte_to_unicode();
        let unicode_to_byte = build_unicode_to_byte();

        Ok(BpeTokenizer {
            vocab,
            token_to_id,
            merge_ranks,
            special_tokens,
            bos_id,
            eos_id,
            chat_template,
            pretokenize_re,
            byte_to_unicode,
            unicode_to_byte,
        })
    }

    /// Encode text into token IDs using byte-level BPE with pretokenization.
    ///
    /// The text is first split into chunks using a regex pattern (matching
    /// llama.cpp's LLAMA3 pretokenizer, which is used for LFM2 models).
    /// BPE merges are then applied within each chunk independently.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }

        // First, split text at special token boundaries.
        // Special tokens (e.g., <|im_start|>, <|im_end|>) are emitted as single
        // token IDs; the text segments between them are BPE-encoded normally.
        let segments = self.split_special_tokens(text);
        let mut result = Vec::new();
        for segment in &segments {
            match segment {
                Segment::Special(id) => result.push(*id),
                Segment::Text(s) => {
                    // Pretokenize: split into chunks using the LLAMA3/LFM2 regex.
                    let chunks: Vec<&str> = self
                        .pretokenize_re
                        .find_iter(s)
                        .map(|m| m.as_str())
                        .collect();
                    for chunk in &chunks {
                        result.extend(self.bpe_encode_chunk(chunk));
                    }
                }
            }
        }
        result
    }

    /// Split text at special token boundaries, returning alternating
    /// text segments and special token IDs.
    fn split_special_tokens<'a>(&self, text: &'a str) -> Vec<Segment<'a>> {
        if self.special_tokens.is_empty() {
            return vec![Segment::Text(text)];
        }

        let mut segments = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            // Find the earliest special token in the remaining text.
            let mut best: Option<(usize, usize, u32)> = None; // (start, end, id)
            for (tok_str, &tok_id) in &self.special_tokens {
                if let Some(pos) = remaining.find(tok_str.as_str()) {
                    if best.is_none() || pos < best.unwrap().0 {
                        best = Some((pos, pos + tok_str.len(), tok_id));
                    }
                }
            }

            match best {
                Some((start, end, id)) => {
                    if start > 0 {
                        segments.push(Segment::Text(&remaining[..start]));
                    }
                    segments.push(Segment::Special(id));
                    remaining = &remaining[end..];
                }
                None => {
                    segments.push(Segment::Text(remaining));
                    break;
                }
            }
        }

        segments
    }

    /// Apply BPE to a single pretokenized chunk.
    fn bpe_encode_chunk(&self, chunk: &str) -> Vec<u32> {
        if chunk.is_empty() {
            return vec![];
        }

        // Convert each raw byte through the GPT-2 byte-to-unicode mapping,
        // then split into individual unicode characters. This ensures space (0x20)
        // becomes Ġ (U+0120), matching how vocab tokens are stored.
        let unicode_str = bytes_to_gpt2_unicode(chunk.as_bytes(), &self.byte_to_unicode);

        // Each mapped unicode character becomes an initial BPE symbol.
        // Characters may be multi-byte in UTF-8 (e.g., Ġ = \xC4\xA0).
        let mut tokens: Vec<Vec<u8>> = unicode_str
            .chars()
            .map(|c| {
                let mut buf = [0u8; 4];
                let s = c.encode_utf8(&mut buf);
                s.as_bytes().to_vec()
            })
            .collect();

        // Repeatedly merge the highest-priority pair
        loop {
            if tokens.len() < 2 {
                break;
            }

            let mut best_rank = usize::MAX;
            let mut best_idx = 0;

            for i in 0..tokens.len() - 1 {
                let pair = (tokens[i].clone(), tokens[i + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = i;
                    }
                }
            }

            if best_rank == usize::MAX {
                break;
            }

            let merged = [tokens[best_idx].as_slice(), tokens[best_idx + 1].as_slice()].concat();
            tokens[best_idx] = merged;
            tokens.remove(best_idx + 1);
        }

        // Convert token byte sequences to IDs
        tokens
            .iter()
            .map(|t| self.token_to_id.get(t).copied().unwrap_or(0))
            .collect()
    }

    /// Decode token IDs back to a string.
    ///
    /// Reverses the GPT-2 byte-to-unicode mapping: collects the unicode chars
    /// from each token's vocab entry, maps them back to raw bytes, then
    /// interprets the result as UTF-8.
    pub fn decode(&self, token_ids: &[u32]) -> String {
        let mut raw_bytes = Vec::new();
        for &id in token_ids {
            if let Some(token_bytes) = self.vocab.get(id as usize) {
                // Try to parse as UTF-8 and reverse the GPT-2 byte-to-unicode mapping.
                // For raw byte tokens (e.g., <0x80> → [0x80], not valid UTF-8),
                // emit the bytes directly — they're already the raw values we want.
                match std::str::from_utf8(token_bytes) {
                    Ok(s) => {
                        for ch in s.chars() {
                            if let Some(&b) = self.unicode_to_byte.get(&ch) {
                                raw_bytes.push(b);
                            } else {
                                let mut buf = [0u8; 4];
                                let encoded = ch.encode_utf8(&mut buf);
                                raw_bytes.extend_from_slice(encoded.as_bytes());
                            }
                        }
                    }
                    Err(_) => {
                        // Non-UTF-8 token (raw byte token) — emit as-is
                        raw_bytes.extend_from_slice(token_bytes);
                    }
                }
            }
        }
        String::from_utf8_lossy(&raw_bytes).into_owned()
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// BOS token ID.
    pub fn bos_token(&self) -> Option<u32> {
        self.bos_id
    }

    /// EOS token ID.
    pub fn eos_token(&self) -> Option<u32> {
        self.eos_id
    }

    /// Get the chat template string if present.
    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    /// Look up a special token by name.
    pub fn special_token_id(&self, name: &str) -> Option<u32> {
        self.special_tokens.get(name).copied()
    }

    /// Check if a token ID is a special/control token.
    pub fn is_special_token(&self, id: u32) -> bool {
        self.special_tokens.values().any(|&v| v == id)
    }
}

// ── Chat template rendering ─────────────────────────────────────────────────

/// A chat message with role and content.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Render a chat template using minijinja.
///
/// The template is a Jinja2 template from the GGUF metadata.
/// Variables available: `messages` (array of {role, content}),
/// `bos_token`, `eos_token`, `add_generation_prompt`.
pub fn apply_chat_template(
    tokenizer: &BpeTokenizer,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
) -> Result<String> {
    let template_str = tokenizer
        .chat_template()
        .context("model has no chat template")?;

    let mut env = minijinja::Environment::new();
    env.add_template("chat", template_str)
        .context("invalid chat template")?;

    let tmpl = env.get_template("chat").unwrap();

    // Build BOS/EOS token strings for the template
    let bos_token = tokenizer
        .bos_token()
        .and_then(|id| tokenizer.vocab.get(id as usize))
        .map(|b| String::from_utf8_lossy(b).into_owned())
        .unwrap_or_default();

    let eos_token = tokenizer
        .eos_token()
        .and_then(|id| tokenizer.vocab.get(id as usize))
        .map(|b| String::from_utf8_lossy(b).into_owned())
        .unwrap_or_default();

    let ctx = minijinja::context! {
        messages => messages,
        bos_token => bos_token,
        eos_token => eos_token,
        add_generation_prompt => add_generation_prompt,
    };

    tmpl.render(ctx).context("rendering chat template")
}

// ── GPT-2 byte-to-unicode mapping ──────────────────────────────────────────

/// Build the GPT-2 byte-to-unicode mapping table.
///
/// GPT-2 BPE uses a reversible mapping from bytes (0-255) to Unicode characters
/// so that every byte can be represented as a valid UTF-8 token. Printable ASCII
/// and Latin-1 bytes map to themselves; other bytes map to Unicode codepoints
/// starting at U+0100. Space (0x20) → Ġ (U+0120).
fn build_byte_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut n = 0u32; // counter for non-printable bytes

    for b in 0u16..256 {
        let ch = match b {
            // Printable ASCII subset + Latin-1 supplement (these map to themselves)
            0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF => b as u32,
            // Everything else maps to U+0100 + n (sequential assignment)
            _ => {
                let c = 256 + n;
                n += 1;
                c
            }
        };
        table[b as usize] = char::from_u32(ch).unwrap();
    }
    table
}

/// Convert raw bytes to a GPT-2 unicode string using the byte-to-unicode mapping.
fn bytes_to_gpt2_unicode(bytes: &[u8], table: &[char; 256]) -> String {
    bytes.iter().map(|&b| table[b as usize]).collect()
}

/// Build the reverse mapping: GPT-2 unicode char → original byte.
fn build_unicode_to_byte() -> HashMap<char, u8> {
    let table = build_byte_to_unicode();
    table
        .iter()
        .enumerate()
        .map(|(b, &ch)| (ch, b as u8))
        .collect()
}

// ── Pretokenization ────────────────────────────────────────────────────────

/// Build the pretokenizer regex based on the `tokenizer.ggml.pre` type.
///
/// Different model families use different pretokenizer patterns:
/// - "lfm2", "llama3", "llama-v3" → LLAMA3 pattern (case-insensitive contractions,
///   1-3 digit groups, newline handling)
/// - "gpt2" → GPT-2 pattern (simpler, case-sensitive contractions)
/// - Others → defaults to LLAMA3 with a warning
fn build_pretokenize_regex(pre_type: &str) -> Regex {
    let pattern = match pre_type {
        // LLAMA3 pattern — used by LFM2, LLaMA 3, and similar models.
        // Simplified: the original has \s+(?!\S)|\s+ which requires lookahead;
        // we use just \s+ since the earlier alternatives capture space-prefixed words.
        "lfm2" | "llama3" | "llama-v3" => concat!(
            r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])",
            r"|[^\r\n\p{L}\p{N}]?\p{L}+",
            r"|\p{N}{1,3}",
            r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
            r"|\s*[\r\n]+",
            r"|\s+",
        ),
        // GPT-2 pattern — simpler, case-sensitive contractions.
        "gpt2" => concat!(
            r"(?:'s|'t|'re|'ve|'m|'ll|'d)",
            r"| ?\p{L}+",
            r"| ?\p{N}+",
            r"| ?[^\s\p{L}\p{N}]+",
            r"|\s+",
        ),
        // Default to LLAMA3 for unknown types (most general pattern)
        other => {
            tracing::warn!(
                "unknown tokenizer.ggml.pre type '{other}', defaulting to LLAMA3 pretokenizer"
            );
            concat!(
                r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])",
                r"|[^\r\n\p{L}\p{N}]?\p{L}+",
                r"|\p{N}{1,3}",
                r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
                r"|\s*[\r\n]+",
                r"|\s+",
            )
        }
    };

    Regex::new(pattern).expect("invalid pretokenizer regex")
}

// ── Token unescaping ────────────────────────────────────────────────────────

/// Convert a token string from GGUF vocabulary to raw bytes.
///
/// GGUF vocabularies use various escape conventions:
/// - `<0xHH>` for raw byte values
/// - `▁` (U+2581) for space (common in sentencepiece)
/// - Regular UTF-8 strings as-is
fn unescape_token(s: &str) -> Vec<u8> {
    // Handle byte tokens: <0xHH>
    if s.starts_with("<0x") && s.ends_with('>') && s.len() == 6 {
        if let Ok(byte) = u8::from_str_radix(&s[3..5], 16) {
            return vec![byte];
        }
    }

    // Handle sentencepiece space marker
    let s = s.replace('▁', " ");

    s.into_bytes()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_tokenizer() -> BpeTokenizer {
        // Build a minimal tokenizer for testing
        // Vocab: individual bytes + some merged tokens
        let mut vocab: Vec<Vec<u8>> = Vec::new();
        let mut token_to_id: HashMap<Vec<u8>, u32> = HashMap::new();

        // First 256 entries: single bytes
        for b in 0u8..=255 {
            vocab.push(vec![b]);
            token_to_id.insert(vec![b], b as u32);
        }

        // Add some merged tokens
        let merged_tokens = vec![
            (256u32, b"he".to_vec()),
            (257, b"ll".to_vec()),
            (258, b"lo".to_vec()),
            (259, b"hell".to_vec()),
            (260, b"hello".to_vec()),
        ];
        for (id, bytes) in &merged_tokens {
            vocab.push(bytes.clone());
            token_to_id.insert(bytes.clone(), *id);
        }

        // Merges in priority order
        let mut merge_ranks = HashMap::new();
        merge_ranks.insert((b"h".to_vec(), b"e".to_vec()), 0); // h+e -> he
        merge_ranks.insert((b"l".to_vec(), b"l".to_vec()), 1); // l+l -> ll
        merge_ranks.insert((b"l".to_vec(), b"o".to_vec()), 2); // l+o -> lo
        merge_ranks.insert((b"he".to_vec(), b"ll".to_vec()), 3); // he+ll -> hell
        merge_ranks.insert((b"hell".to_vec(), b"o".to_vec()), 4); // hell+o -> hello

        BpeTokenizer {
            vocab,
            token_to_id,
            merge_ranks,
            special_tokens: HashMap::new(),
            bos_id: None,
            eos_id: None,
            chat_template: None,
            pretokenize_re: build_pretokenize_regex("lfm2"),
            byte_to_unicode: build_byte_to_unicode(),
            unicode_to_byte: build_unicode_to_byte(),
        }
    }

    #[test]
    fn test_byte_to_unicode_space() {
        let table = build_byte_to_unicode();
        // Space (0x20) should map to Ġ (U+0120)
        assert_eq!(table[0x20], '\u{0120}');
        // Printable ASCII should map to itself
        assert_eq!(table[b'A' as usize], 'A');
        assert_eq!(table[b'z' as usize], 'z');
        assert_eq!(table[b'0' as usize], '0');
        // Newline (0x0A) should NOT map to itself (it's a control char)
        assert_ne!(table[0x0A], '\n');
    }

    #[test]
    fn test_byte_unicode_roundtrip() {
        let table = build_byte_to_unicode();
        let reverse = build_unicode_to_byte();
        for b in 0u8..=255 {
            let ch = table[b as usize];
            assert_eq!(reverse[&ch], b, "roundtrip failed for byte {b:#04x}");
        }
    }

    #[test]
    fn test_pretokenize_splits_words() {
        let re = build_pretokenize_regex("lfm2");
        let chunks: Vec<&str> = re
            .find_iter("The meaning of life")
            .map(|m| m.as_str())
            .collect();
        assert_eq!(chunks, vec!["The", " meaning", " of", " life"]);
    }

    #[test]
    fn test_pretokenize_contractions() {
        let re = build_pretokenize_regex("lfm2");
        let chunks: Vec<&str> = re.find_iter("I'm don't").map(|m| m.as_str()).collect();
        assert!(chunks.contains(&"'m"));
        assert!(chunks.contains(&"'t"));
    }

    #[test]
    fn test_pretokenize_numbers() {
        let re = build_pretokenize_regex("lfm2");
        let chunks: Vec<&str> = re.find_iter("test 12345").map(|m| m.as_str()).collect();
        // Numbers split into 1-3 digit groups
        assert_eq!(chunks, vec!["test", " ", "123", "45"]);
    }

    /// Build a tokenizer with GPT-2-style vocab (Ġ-prefixed tokens for spaces).
    fn make_gpt2_style_tokenizer() -> BpeTokenizer {
        let table = build_byte_to_unicode();
        let mut vocab: Vec<Vec<u8>> = Vec::new();
        let mut token_to_id: HashMap<Vec<u8>, u32> = HashMap::new();

        // Token 0: pad
        vocab.push(b"<pad>".to_vec());
        token_to_id.insert(b"<pad>".to_vec(), 0);

        // Individual GPT-2 unicode chars for common bytes
        for b in 0u8..=127 {
            let ch = table[b as usize];
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            let bytes = s.as_bytes().to_vec();
            let id = vocab.len() as u32;
            vocab.push(bytes.clone());
            token_to_id.insert(bytes, id);
        }

        // Add merged tokens with Ġ prefix (space = U+0120 in GPT-2 encoding)
        let space_char = table[b' ' as usize]; // Ġ
        let add_token =
            |vocab: &mut Vec<Vec<u8>>, map: &mut HashMap<Vec<u8>, u32>, s: &str| -> u32 {
                let bytes = s.as_bytes().to_vec();
                let id = vocab.len() as u32;
                vocab.push(bytes.clone());
                map.insert(bytes, id);
                id
            };

        // "Hi" merged token
        let _hi_id = add_token(&mut vocab, &mut token_to_id, "Hi");
        // "Ġworld" merged token (space-prefixed "world")
        let space_world = format!("{space_char}world");
        let _world_id = add_token(&mut vocab, &mut token_to_id, &space_world);

        // Merges
        let mut merge_ranks = HashMap::new();
        merge_ranks.insert((b"H".to_vec(), b"i".to_vec()), 0);
        // Ġ + w merge
        let g_bytes = {
            let mut buf = [0u8; 4];
            space_char.encode_utf8(&mut buf).as_bytes().to_vec()
        };
        merge_ranks.insert((g_bytes.clone(), b"w".to_vec()), 1);
        // Ġw + o
        let gw_bytes = [g_bytes.as_slice(), b"w"].concat();
        merge_ranks.insert((gw_bytes.clone(), b"o".to_vec()), 2);
        // Ġwo + r
        let gwor = [gw_bytes.as_slice(), b"o"].concat();
        merge_ranks.insert((gwor.clone(), b"r".to_vec()), 3);
        // Ġwor + l
        let gworl = [gwor.as_slice(), b"r"].concat();
        merge_ranks.insert((gworl.clone(), b"l".to_vec()), 4);
        // Ġworl + d
        let gworld = [gworl.as_slice(), b"l"].concat();
        merge_ranks.insert((gworld.clone(), b"d".to_vec()), 5);

        BpeTokenizer {
            vocab,
            token_to_id,
            merge_ranks,
            special_tokens: HashMap::new(),
            bos_id: None,
            eos_id: None,
            chat_template: None,
            pretokenize_re: build_pretokenize_regex("lfm2"),
            byte_to_unicode: build_byte_to_unicode(),
            unicode_to_byte: build_unicode_to_byte(),
        }
    }

    #[test]
    fn test_gpt2_encode_space_prefix() {
        let tok = make_gpt2_style_tokenizer();
        let ids = tok.encode("Hi world");
        // "Hi world" → pretokenize → ["Hi", " world"]
        // "Hi" → BPE merges to "Hi" token
        // " world" → byte-to-unicode → "Ġworld" → BPE merges to "Ġworld" token
        let hi_id = *tok.token_to_id.get(b"Hi".as_slice()).unwrap();
        let table = build_byte_to_unicode();
        let space_world = format!("{}world", table[b' ' as usize]);
        let world_id = *tok.token_to_id.get(space_world.as_bytes()).unwrap();
        assert_eq!(ids, vec![hi_id, world_id]);
    }

    #[test]
    fn test_gpt2_decode_reverses_encode() {
        let tok = make_gpt2_style_tokenizer();
        let ids = tok.encode("Hi world");
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "Hi world");
    }

    #[test]
    fn test_encode_single_bytes() {
        let tok = make_test_tokenizer();
        let ids = tok.encode("ab");
        assert_eq!(ids, vec![b'a' as u32, b'b' as u32]);
    }

    #[test]
    fn test_encode_with_merges() {
        let tok = make_test_tokenizer();
        let ids = tok.encode("hello");
        // Should merge: h+e->he, l+l->ll, he+ll->hell, hell+o->hello
        assert_eq!(ids, vec![260]); // "hello" as single token
    }

    #[test]
    fn test_encode_partial_merges() {
        let tok = make_test_tokenizer();
        let ids = tok.encode("hell");
        // h+e->he, l+l->ll, he+ll->hell
        assert_eq!(ids, vec![259]); // "hell" as single token
    }

    #[test]
    fn test_decode_roundtrip() {
        let tok = make_test_tokenizer();
        let text = "hello";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_decode_single_bytes() {
        let tok = make_test_tokenizer();
        let decoded = tok.decode(&[72, 105]); // 'H', 'i'
        assert_eq!(decoded, "Hi");
    }

    #[test]
    fn test_encode_empty() {
        let tok = make_test_tokenizer();
        assert_eq!(tok.encode(""), vec![]);
    }

    #[test]
    fn test_unescape_byte_token() {
        assert_eq!(unescape_token("<0x0A>"), vec![0x0A]); // newline
        assert_eq!(unescape_token("<0xFF>"), vec![0xFF]);
        assert_eq!(unescape_token("<0x00>"), vec![0x00]);
    }

    #[test]
    fn test_unescape_space_marker() {
        assert_eq!(unescape_token("▁hello"), b" hello");
        assert_eq!(unescape_token("▁"), b" ");
    }

    #[test]
    fn test_unescape_regular() {
        assert_eq!(unescape_token("hello"), b"hello");
    }

    #[test]
    fn test_chat_template_rendering() {
        let mut tok = make_test_tokenizer();
        tok.chat_template = Some(
            "{% for msg in messages %}{{ msg.role }}: {{ msg.content }}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
                .to_string(),
        );

        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello!".to_string(),
        }];

        let result = apply_chat_template(&tok, &messages, true).unwrap();
        assert_eq!(result, "user: Hello!\nassistant: ");
    }
}
