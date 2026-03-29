use std::collections::HashMap;

/// A minimal byte-level BPE tokenizer.
#[allow(dead_code)]
pub struct BpeTokenizer {
    pub(crate) vocab: Vec<String>,
    pub(crate) merges: Vec<(String, String)>,
    pub(crate) token_to_id: HashMap<String, u32>,
    pub(crate) special_tokens: HashMap<String, u32>,
}

impl BpeTokenizer {
    /// Encode text into token IDs.
    pub fn encode(&self, _text: &str) -> Vec<u32> {
        todo!("BPE encoding not yet implemented")
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, _tokens: &[u32]) -> String {
        todo!("BPE decoding not yet implemented")
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Beginning-of-sequence token ID.
    pub fn bos_token(&self) -> Option<u32> {
        self.special_tokens.get("<|begin_of_text|>").copied()
    }

    /// End-of-sequence token ID.
    pub fn eos_token(&self) -> Option<u32> {
        self.special_tokens.get("<|end_of_text|>").copied()
    }
}
