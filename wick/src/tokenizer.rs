use std::collections::HashMap;

use anyhow::{Context, Result};

use crate::gguf::{GgufFile, GgufValue};

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

        Ok(BpeTokenizer {
            vocab,
            token_to_id,
            merge_ranks,
            special_tokens,
            bos_id,
            eos_id,
            chat_template,
        })
    }

    /// Encode text into token IDs using byte-level BPE.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }

        // Start with each byte as its own token
        let bytes = text.as_bytes();
        let mut tokens: Vec<Vec<u8>> = bytes.iter().map(|&b| vec![b]).collect();

        // Repeatedly merge the highest-priority pair
        loop {
            if tokens.len() < 2 {
                break;
            }

            // Find the merge pair with the lowest rank (= highest priority)
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
                break; // No more merges possible
            }

            // Apply the merge: combine tokens[best_idx] and tokens[best_idx + 1]
            let merged = [tokens[best_idx].as_slice(), tokens[best_idx + 1].as_slice()].concat();
            tokens[best_idx] = merged;
            tokens.remove(best_idx + 1);
        }

        // Convert token byte sequences to IDs
        tokens
            .iter()
            .map(|t| {
                self.token_to_id.get(t).copied().unwrap_or(0) // fallback to token 0 for unknown sequences
            })
            .collect()
    }

    /// Decode token IDs back to a string.
    pub fn decode(&self, token_ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in token_ids {
            if let Some(token_bytes) = self.vocab.get(id as usize) {
                bytes.extend_from_slice(token_bytes);
            }
        }
        // Best-effort UTF-8 decode, replacing invalid sequences
        String::from_utf8_lossy(&bytes).into_owned()
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
        }
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
