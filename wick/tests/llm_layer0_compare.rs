#![cfg(all(feature = "metal", target_os = "macos"))]

/// Standalone test: run ONLY LLM layer 0 (a GatedConv block) with a known token
/// embedding and dump intermediates for comparison with a Python reference.

fn dump(path: &str, data: &[f32]) {
    let bytes: &[u8] = bytemuck::cast_slice(data);
    std::fs::write(path, bytes).unwrap();
}

fn stat(label: &str, data: &[f32]) {
    let rms = (data.iter().map(|x| x * x).sum::<f32>() / data.len() as f32).sqrt();
    eprintln!(
        "[llm_l0] {label}: n={} rms={rms:.8} first3=[{:.8}, {:.8}, {:.8}]",
        data.len(),
        data[0],
        data[1],
        data[2]
    );
}

#[test]
fn llm_layer0_conv_standalone() {
    use wick::backend::cpu;
    use wick::model::Model;
    use wick::model::lfm2::Lfm2Model;

    let model_path = std::path::PathBuf::from(std::env::var("HOME").expect("HOME not set"))
        .join(".leap/models/LFM2.5-Audio-1.5B-Q4_0/LFM2.5-Audio-1.5B-Q4_0.gguf");
    if !model_path.exists() {
        eprintln!("model not found, skipping");
        return;
    }

    let gguf = wick::gguf::GgufFile::open(&model_path).unwrap();
    let model = Lfm2Model::from_gguf(gguf).unwrap();
    let cfg = model.config();
    let hs = cfg.hidden_size;

    // Token 1 = <|im_start|> (first prompt token)
    let hidden_orig = model.dequantize_embedding(1);
    assert_eq!(hidden_orig.len(), hs);
    stat("embedding(tok=1)", &hidden_orig);
    dump("/tmp/wick_llm_l0_embedding.bin", &hidden_orig);

    // -- Layer 0: GatedConv block --
    let mut hidden = hidden_orig.clone();

    // 1. RMSnorm
    let mut normed = hidden.clone();
    cpu::rmsnorm(&mut normed, model.attn_norm_weight(0), cfg.rms_norm_eps);
    stat("normed", &normed);
    dump("/tmp/wick_llm_l0_normed.bin", &normed);

    // 2. in_proj → 3*hidden
    let mut proj = vec![0.0f32; 3 * hs];
    model.conv_in_proj_gemv(0, &normed, &mut proj);
    stat("in_proj", &proj);
    dump("/tmp/wick_llm_l0_in_proj.bin", &proj);

    // Split: b, c, x
    let b = &proj[..hs];
    let c = &proj[hs..2 * hs];
    let x = &proj[2 * hs..];
    stat("b", b);
    stat("c", c);
    stat("x", x);

    // 3. bx = b * x
    let bx: Vec<f32> = b.iter().zip(x).map(|(bi, xi)| bi * xi).collect();
    stat("bx", &bx);

    // 4. Conv1d (first token: conv state is zero, only last kernel weight)
    let kernel_size = cfg.conv_kernel_size.unwrap_or(3);
    let d_conv = kernel_size - 1;
    let conv_w = model.conv_weight(0).unwrap();
    let mut conv_out = vec![0.0f32; hs];
    // Full conv (buffer is zero):
    for ch in 0..hs {
        let mut sum = 0.0f32;
        // d_conv slots are zero, only current input contributes
        sum += bx[ch] * conv_w[ch * kernel_size + d_conv];
        conv_out[ch] = sum;
    }
    stat("conv_out", &conv_out);

    // 5. gated = c * conv_out
    let gated: Vec<f32> = c.iter().zip(&conv_out).map(|(ci, co)| ci * co).collect();
    stat("gated", &gated);

    // 6. out_proj
    let mut block_out = vec![0.0f32; hs];
    model.conv_out_proj_gemv(0, &gated, &mut block_out);
    stat("out_proj", &block_out);

    // 7. Residual
    for (h, o) in hidden.iter_mut().zip(&block_out) {
        *h += o;
    }
    stat("after_residual", &hidden);
    dump("/tmp/wick_llm_l0_after_conv_residual.bin", &hidden);

    // 8. FFN: rmsnorm → gate + up → silu_mul → down → residual
    let ffn_residual = hidden.clone();
    cpu::rmsnorm(&mut hidden, model.ffn_norm_weight(0), cfg.rms_norm_eps);
    stat("ffn_normed", &hidden);

    let is = cfg.intermediate_size;
    let mut gate = vec![0.0f32; is];
    let mut up = vec![0.0f32; is];
    model.ffn_gate_gemv(0, &hidden, &mut gate);
    model.ffn_up_gemv(0, &hidden, &mut up);
    cpu::silu_mul_inplace(&mut gate, &up);
    stat("ffn_silu_mul", &gate);

    let mut down = vec![0.0f32; hs];
    model.ffn_down_gemv(0, &gate, &mut down);

    let output: Vec<f32> = ffn_residual.iter().zip(&down).map(|(r, d)| r + d).collect();
    stat("FINAL", &output);
    dump("/tmp/wick_llm_l0_output.bin", &output);

    // Compare with the run_layers dump
    if std::path::Path::new("/tmp/wick_llm_layer0.bin").exists() {
        let run_layers_l0: Vec<f32> = std::fs::read("/tmp/wick_llm_layer0.bin")
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let n = output.len().min(run_layers_l0.len());
        let exact = (0..n)
            .filter(|&i| output[i].to_bits() == run_layers_l0[i].to_bits())
            .count();
        let cos: f64 = {
            let dot: f64 = output
                .iter()
                .zip(&run_layers_l0)
                .map(|(a, b)| (*a as f64) * (*b as f64))
                .sum();
            let na: f64 = output
                .iter()
                .map(|a| (*a as f64) * (*a as f64))
                .sum::<f64>()
                .sqrt();
            let nb: f64 = run_layers_l0
                .iter()
                .map(|a| (*a as f64) * (*a as f64))
                .sum::<f64>()
                .sqrt();
            dot / (na * nb)
        };
        eprintln!("[llm_l0] vs run_layers: cosine={cos:.10} exact={exact}/{n}");
    }
}
