#![cfg(all(feature = "metal", target_os = "macos"))]

/// Standalone test: run ONLY depthformer layer 0 with a known input and dump
/// intermediates + final output for comparison with ggml reference values.

fn load_f32_bin(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn dump_f32_bin(path: &str, data: &[f32]) {
    let bytes: &[u8] = bytemuck::cast_slice(data);
    std::fs::write(path, bytes).unwrap_or_else(|e| panic!("cannot write {path}: {e}"));
}

fn rms_stat(data: &[f32]) -> f32 {
    (data.iter().map(|x| x * x).sum::<f32>() / data.len() as f32).sqrt()
}

/// Apply RoPE (interleaved / LLAMA_ROPE_TYPE_NORM) to a single head.
/// Exact copy of the private function in audio_decoder.rs.
fn apply_rope_interleaved(x: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
    let theta_scale = freq_base.powf(-2.0 / head_dim as f32);
    let mut theta = pos as f32;
    for i in 0..head_dim / 2 {
        let (sin_t, cos_t) = theta.sin_cos();
        let x0 = x[2 * i];
        let x1 = x[2 * i + 1];
        x[2 * i] = x0 * cos_t - x1 * sin_t;
        x[2 * i + 1] = x0 * sin_t + x1 * cos_t;
        theta *= theta_scale;
    }
}

#[test]
fn depthformer_layer0_standalone() {
    use wick::backend::cpu;

    // ── Load model ──────────────────────────────────────────────────────
    let vocoder_path = std::path::PathBuf::from(std::env::var("HOME").expect("HOME not set"))
        .join(".leap/models/LFM2.5-Audio-1.5B-Q4_0/vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf");
    if !vocoder_path.exists() {
        eprintln!("vocoder GGUF not found, skipping");
        return;
    }
    let voc_gguf = wick::gguf::GgufFile::open(&vocoder_path).unwrap();
    let dw = wick::model::audio_decoder::AudioDecoderWeights::from_gguf(&voc_gguf).unwrap();

    // ── Load known input ────────────────────────────────────────────────
    let input_path = "/tmp/wick_df_input.bin";
    if !std::path::Path::new(input_path).exists() {
        eprintln!("{input_path} not found, skipping");
        return;
    }
    let input = load_f32_bin(input_path);
    assert_eq!(input.len(), 1024, "expected 1024 f32s in input");

    let cfg = &dw.depthformer_config;
    let n_embd = cfg.n_embd;
    let n_head = cfg.n_head;
    let n_kv = cfg.n_head_kv;
    let hd = cfg.n_embd_head;
    let pos: usize = 0;

    assert_eq!(input.len(), n_embd);

    let lw = &dw.depthformer_layers[0]; // layer 0 only

    let mut cur = input.clone();
    let residual = cur.clone();

    // ── 1. RMSnorm ──────────────────────────────────────────────────────
    cpu::rmsnorm(&mut cur, &lw.operator_norm, cfg.rms_norm_eps);
    eprintln!(
        "[layer0] after rmsnorm: rms={:.8} first5={:.8?}",
        rms_stat(&cur),
        &cur[..5]
    );
    dump_f32_bin("/tmp/wick_layer0_after_rmsnorm.bin", &cur);

    // ── 2. Fused QKV projection ─────────────────────────────────────────
    let qkv_dim = (n_head + 2 * n_kv) * hd;
    let mut qkv = vec![0.0f32; qkv_dim];
    lw.wqkv.gemv(&cur, &mut qkv);

    let q_dim = n_head * hd;
    let k_dim = n_kv * hd;
    let mut q = qkv[..q_dim].to_vec();
    let mut k = qkv[q_dim..q_dim + k_dim].to_vec();
    let v = qkv[q_dim + k_dim..].to_vec();

    eprintln!(
        "[layer0] after QKV: q_rms={:.8} k_rms={:.8} v_rms={:.8}",
        rms_stat(&q),
        rms_stat(&k),
        rms_stat(&v)
    );
    dump_f32_bin("/tmp/wick_layer0_after_qkv.bin", &qkv);

    // ── 3. Per-head RMSnorm on Q and K ──────────────────────────────────
    for h in 0..n_head {
        let s = &mut q[h * hd..(h + 1) * hd];
        cpu::rmsnorm(s, &lw.q_norm, cfg.rms_norm_eps);
    }
    for h in 0..n_kv {
        let s = &mut k[h * hd..(h + 1) * hd];
        cpu::rmsnorm(s, &lw.k_norm, cfg.rms_norm_eps);
    }

    // ── 4. RoPE on Q and K (interleaved) ────────────────────────────────
    for h in 0..n_head {
        apply_rope_interleaved(&mut q[h * hd..(h + 1) * hd], pos, hd, cfg.rope_freq_base);
    }
    for h in 0..n_kv {
        apply_rope_interleaved(&mut k[h * hd..(h + 1) * hd], pos, hd, cfg.rope_freq_base);
    }

    // ── 5. Write K, V to fresh cache (pos=0) ───────────────────────────
    // Single-layer cache: [max_seq × n_kv × hd]
    let cache_size = cfg.max_seq_len * n_kv * hd;
    let mut cache_k = vec![0.0f32; cache_size];
    let mut cache_v = vec![0.0f32; cache_size];
    let kv_dim = n_kv * hd;

    for h in 0..n_kv {
        let cache_off = pos * kv_dim + h * hd;
        cache_k[cache_off..cache_off + hd].copy_from_slice(&k[h * hd..(h + 1) * hd]);
        cache_v[cache_off..cache_off + hd].copy_from_slice(&v[h * hd..(h + 1) * hd]);
    }

    // ── 6. Attention: Q×K scores → softmax → V accumulation ────────────
    let seq_len = pos + 1; // = 1
    let group_size = n_head / n_kv;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut attn_out = vec![0.0f32; n_head * hd];

    for h in 0..n_head {
        let kv_h = h / group_size;
        let q_head = &q[h * hd..(h + 1) * hd];
        let kv_h_offset = kv_h * hd;

        let mut scores = vec![0.0f32; seq_len];
        cpu::attn_scores(
            q_head,
            &cache_k,
            &mut scores,
            kv_dim,
            kv_h_offset,
            hd,
            scale,
            seq_len,
        );
        cpu::softmax_inplace(&mut scores);

        let out = &mut attn_out[h * hd..(h + 1) * hd];
        cpu::attn_values(&scores, &cache_v, out, kv_dim, kv_h_offset, hd, seq_len);
    }

    eprintln!(
        "[layer0] after attention: rms={:.8} first5={:.8?}",
        rms_stat(&attn_out),
        &attn_out[..5]
    );
    dump_f32_bin("/tmp/wick_layer0_after_attn.bin", &attn_out);

    // ── 7. Output projection + residual ─────────────────────────────────
    let mut proj = vec![0.0f32; n_embd];
    lw.wo.gemv(&attn_out, &mut proj);
    cur = residual.iter().zip(&proj).map(|(r, p)| r + p).collect();

    // ── 8. FFN: RMSnorm → SwiGLU(w1, w3) → w2 → residual ──────────────
    let ffn_residual = cur.clone();
    cpu::rmsnorm(&mut cur, &lw.ffn_norm, cfg.rms_norm_eps);

    let mut gate = vec![0.0f32; cfg.ffn_dim];
    let mut up = vec![0.0f32; cfg.ffn_dim];
    lw.w1.gemv(&cur, &mut gate);
    lw.w3.gemv(&cur, &mut up);
    cpu::silu_mul_inplace(&mut gate, &up);

    let mut down = vec![0.0f32; n_embd];
    lw.w2.gemv(&gate, &mut down);
    cur = ffn_residual.iter().zip(&down).map(|(r, d)| r + d).collect();

    eprintln!(
        "[layer0] after FFN (final): rms={:.8} first5={:.8?}",
        rms_stat(&cur),
        &cur[..5]
    );
    dump_f32_bin("/tmp/wick_layer0_after_ffn.bin", &cur);

    // ── Dump final layer 0 output ───────────────────────────────────────
    dump_f32_bin("/tmp/wick_layer0_standalone.bin", &cur);
    eprintln!("[layer0] all outputs dumped to /tmp/wick_layer0_*.bin");
}
