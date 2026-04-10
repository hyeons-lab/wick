#[test]
fn load_vocoder_gguf() {
    let path = std::path::PathBuf::from(std::env::var("HOME").expect("HOME not set"))
        .join(".leap/models/LFM2.5-Audio-1.5B-Q4_0/vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf");
    if !path.exists() {
        eprintln!("skipping — vocoder not found");
        return;
    }

    let gguf = wick::gguf::GgufFile::open(&path).unwrap();
    let weights = wick::model::audio_decoder::AudioDecoderWeights::from_gguf(&gguf).unwrap();

    let dc = &weights.depthformer_config;
    eprintln!(
        "Depthformer: {}L, embd={}, head={}, kv={}, hd={}, ffn={}",
        dc.n_layer, dc.n_embd, dc.n_head, dc.n_head_kv, dc.n_embd_head, dc.ffn_dim
    );

    let dec = &weights.decoder_config;
    eprintln!(
        "Decoder: {}cb, vocab={}, embd={}",
        dec.n_codebook, dec.n_vocab, dec.n_embd
    );

    assert_eq!(dc.n_layer, 6);
    assert_eq!(dc.n_embd, 1024);
    assert_eq!(dc.n_embd_head, 32);
    assert_eq!(dec.n_codebook, 8);
    assert_eq!(dec.n_vocab, 2049);
    assert_eq!(weights.depthformer_layers.len(), 6);
    assert_eq!(weights.depth_embeddings.len(), 8);
    eprintln!("Vocoder load OK");
}

#[test]
fn load_detokenizer() {
    let path = std::path::PathBuf::from(std::env::var("HOME").expect("HOME not set"))
        .join(".leap/models/LFM2.5-Audio-1.5B-Q4_0/vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf");
    if !path.exists() {
        eprintln!("skipping — vocoder not found");
        return;
    }

    let gguf = wick::gguf::GgufFile::open(&path).unwrap();
    let detok = wick::model::audio_decoder::DetokenizerWeights::from_gguf(&gguf).unwrap();

    let c = &detok.config;
    eprintln!(
        "Detokenizer: {}L, embd={}, head={}/{}, ffn={}, n_fft={}, sr={}",
        c.n_layer, c.n_embd, c.n_head, c.n_head_kv, c.ffn_dim, c.n_fft, c.sample_rate
    );

    assert_eq!(c.n_layer, 8);
    assert_eq!(c.n_embd, 512);
    assert_eq!(c.n_head, 16);
    assert_eq!(c.n_head_kv, 8);
    assert_eq!(c.n_embd_head, 32);
    assert_eq!(detok.layers.len(), 8);
    assert_eq!(detok.lin_b.len(), 1282); // n_fft/2 + 1 = 641, × 2 (real/imag) = 1282
    eprintln!("Detokenizer load OK");
}
