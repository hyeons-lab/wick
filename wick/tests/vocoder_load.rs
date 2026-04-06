#[test]
fn load_vocoder_gguf() {
    let path = std::path::Path::new(env!("HOME"))
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
