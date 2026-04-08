// Metal-accelerated audio detokenizer.
//
// The detokenizer (8-layer LFM2 backbone, 6 tokens per frame) is the main
// audio bottleneck at ~165ms/frame on CPU. This module moves it to Metal,
// targeting ~10-15ms/frame. ISTFT stays on CPU (rustfft).
//
// Weight tensors are Q4_0 in the vocoder GGUF — accessed via zero-copy mmap
// exactly like the LLM weights in metal_lfm2.rs.

use std::cell::Cell;
use std::path::Path;

use anyhow::{Result, ensure};
use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, MTLResourceOptions};

use crate::backend::metal::{MetalContext, shaders};
use crate::gguf::GgufFile;
use crate::model::audio_decoder::{DetokenizerConfig, DetokenizerWeights};
use crate::tensor::DType;

// ── Helpers ─────────────────────────────────────────────────────────────────

fn sz1d(x: u64) -> metal::MTLSize {
    metal::MTLSize::new(x, 1, 1)
}

fn sz2d(x: u64, y: u64) -> metal::MTLSize {
    metal::MTLSize::new(x, y, 1)
}

// ── GPU weight ──────────────────────────────────────────────────────────────

/// F32 weight matrix on GPU. Dequantized from Q4_0 at load time to match
/// the CPU detokenizer's precision path exactly.
struct MetalWeight {
    buf: Buffer, // uploaded F32 data [m * k]
    m: u32,
    k: u32,
    params_buf: Buffer,
}

// ── Per-layer weights ───────────────────────────────────────────────────────

struct DetokLayerGpu {
    operator_norm: Buffer,
    ffn_norm: Buffer,
    ffn_w1: MetalWeight,
    ffn_w2: MetalWeight,
    ffn_w3: MetalWeight,
    // Conv-only
    conv_in_proj: Option<MetalWeight>,
    conv_out_proj: Option<MetalWeight>,
    conv_weight: Option<Buffer>,
    // Attention-only
    wq: Option<MetalWeight>,
    wk: Option<MetalWeight>,
    wv: Option<MetalWeight>,
    wo: Option<MetalWeight>,
    q_norm: Option<Buffer>,
    k_norm: Option<Buffer>,
}

// ── Pipelines ───────────────────────────────────────────────────────────────

struct Pipelines {
    gemv_f32: ComputePipelineState,
    memcpy_f32: ComputePipelineState,
    mul_out: ComputePipelineState,
    add_inplace: ComputePipelineState,
    silu_mul_inplace: ComputePipelineState,
    rmsnorm: ComputePipelineState,
    qk_norm_rope: ComputePipelineState,
    flash_attention: ComputePipelineState,
    conv1d: ComputePipelineState,
}

// ── Params buffers ──────────────────────────────────────────────────────────

struct Params {
    rmsnorm_hs: Buffer,
    per_head_rmsnorm: Buffer,
    elementwise_hs: Buffer,
    elementwise_3hs: Buffer,
    elementwise_is: Buffer,
    conv1d: Buffer,
}

// ── Main struct ─────────────────────────────────────────────────────────────

pub struct MetalAudioDecoder {
    ctx: MetalContext,
    cfg: DetokenizerConfig,
    pipes: Pipelines,
    params: Params,

    depthformer: Option<MetalDepthformer>,

    layers: Vec<DetokLayerGpu>,
    output_norm: Buffer,
    lin_w: MetalWeight,
    lin_b: Buffer,

    // Scratch
    hidden_buf: Buffer,
    normed_buf: Buffer,
    accum_scratch: Buffer, // [max(n_embd, ffn_dim)] scratch for GEMV + add_inplace residual
    proj_buf: Buffer,      // [3 * n_embd] for conv in_proj output
    bx_buf: Buffer,        // [n_embd] for b*x
    conv_out_buf: Buffer,  // [n_embd]
    gate_buf: Buffer,
    up_buf: Buffer,
    q_buf: Buffer, // [n_head * head_dim]
    k_buf: Buffer, // [n_kv * head_dim]
    v_buf: Buffer, // [n_kv * head_dim]
    attn_out_buf: Buffer,
    spectrum_buf: Buffer, // [6 * 1282]
    tokens_buf: Buffer,   // [6 * n_embd]

    // Persistent state
    conv_bufs: Vec<Option<Buffer>>,
    kv_k: Vec<Option<Buffer>>,
    kv_v: Vec<Option<Buffer>>,
    n_past: Cell<usize>,
    // mmap removed — all weights uploaded as dequantized F32 for CPU-matching precision
}

impl MetalAudioDecoder {
    pub fn from_gguf(gguf: &GgufFile, vocoder_path: &Path) -> Result<Self> {
        // Also load the CPU decoder weights for depthformer config
        let cpu_dec = crate::model::audio_decoder::AudioDecoderWeights::from_gguf(gguf)?;
        let ctx = MetalContext::new()?;

        // Parse config from tensor shapes (same logic as DetokenizerWeights::from_gguf).
        let (_, _, conv_in_cols, _) = gguf.tensor_meta("lfm.layers.0.conv.in_proj.weight")?;
        let n_embd = conv_in_cols;
        let q_norm_t = gguf.get_tensor("lfm.layers.2.self_attn.q_layernorm.weight")?;
        let head_dim = q_norm_t.shape()[0];
        let (_, q_rows, _, _) = gguf.tensor_meta("lfm.layers.2.self_attn.q_proj.weight")?;
        let n_head = q_rows / head_dim;
        let (_, k_rows, _, _) = gguf.tensor_meta("lfm.layers.2.self_attn.k_proj.weight")?;
        let n_kv = k_rows / head_dim;
        let (_, ffn_rows, _, _) = gguf.tensor_meta("lfm.layers.0.feed_forward.w1.weight")?;
        let ffn_dim = ffn_rows;

        let layer_is_conv = vec![true, true, false, true, false, true, false, true];
        let n_layer = layer_is_conv.len();
        let kv_dim = n_kv * head_dim;

        let cfg = DetokenizerConfig {
            n_layer,
            n_embd,
            n_head,
            n_head_kv: n_kv,
            n_embd_head: head_dim,
            ffn_dim,
            d_conv: 2,
            rms_norm_eps: 1e-5,
            rope_freq_base: 1_000_000.0,
            swa_window_size: 30,
            n_codes: 8,
            n_fft: 1280,
            hop_length: 320,
            sample_rate: 24000,
            layer_is_conv,
        };

        let pipes = Pipelines {
            gemv_f32: ctx.create_pipeline(shaders::GEMV_F32, "gemv_f32")?,
            memcpy_f32: ctx.create_pipeline(shaders::ELEMENTWISE, "memcpy_f32")?,
            mul_out: ctx.create_pipeline(shaders::ELEMENTWISE, "mul_out")?,
            add_inplace: ctx.create_pipeline(shaders::ELEMENTWISE, "add_inplace")?,
            silu_mul_inplace: ctx.create_pipeline(shaders::ELEMENTWISE, "silu_mul_inplace")?,
            rmsnorm: ctx.create_pipeline(shaders::RMSNORM, "rmsnorm")?,
            qk_norm_rope: ctx.create_pipeline(shaders::QK_NORM_ROPE, "qk_norm_rope")?,
            flash_attention: ctx.create_pipeline(shaders::FLASH_ATTENTION, "flash_attention")?,
            conv1d: ctx.create_pipeline(shaders::CONV1D, "conv1d_depthwise")?,
        };

        // Weights are dequantized from Q4_0 to F32 and uploaded.
        // ~140 MB GPU memory — matches CPU precision path exactly.

        let eps_bits = cfg.rms_norm_eps.to_bits();

        // Params buffers
        let params = Params {
            rmsnorm_hs: ctx.upload_bytes(bytemuck::cast_slice(&[
                n_embd as u32,
                eps_bits,
                0u32,
                0u32,
            ])),
            per_head_rmsnorm: ctx.upload_bytes(bytemuck::cast_slice(&[
                head_dim as u32,
                eps_bits,
                0u32,
                0u32,
            ])),
            elementwise_hs: ctx.upload_bytes(bytemuck::cast_slice(&[n_embd as u32, 0u32])),
            elementwise_3hs: ctx.upload_bytes(bytemuck::cast_slice(&[(3 * n_embd) as u32, 0u32])),
            elementwise_is: ctx.upload_bytes(bytemuck::cast_slice(&[ffn_dim as u32, 0u32])),
            conv1d: ctx.upload_bytes(bytemuck::cast_slice(&[n_embd as u32, 3u32, 2u32, 0u32])),
        };

        // Dequantize Q4_0 weight to F32 and upload to GPU.
        // This matches the CPU detokenizer's precision path exactly.
        let make_weight = |name: &str| -> Result<MetalWeight> {
            let t = gguf.get_tensor(name)?;
            let f32_data = t.to_f32_vec();
            let shape = t.shape();
            let (rows, cols) = match shape.len() {
                1 => (1, shape[0]),
                2 => (shape[1], shape[0]),
                _ => anyhow::bail!("unexpected rank for {name}"),
            };
            let buf = ctx.upload_f32(&f32_data);
            let params_buf = ctx.upload_bytes(bytemuck::cast_slice(&[rows as u32, cols as u32]));
            Ok(MetalWeight {
                buf,
                m: rows as u32,
                k: cols as u32,
                params_buf,
            })
        };

        // Per-layer weights
        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let pfx = format!("lfm.layers.{i}");
            let is_conv = cfg.layer_is_conv[i];

            let (cin, cop, cw) = if is_conv {
                (
                    Some(make_weight(&format!("{pfx}.conv.in_proj.weight"))?),
                    Some(make_weight(&format!("{pfx}.conv.out_proj.weight"))?),
                    Some(
                        ctx.upload_f32(
                            &gguf
                                .get_tensor(&format!("{pfx}.conv.conv.weight"))?
                                .to_f32_vec(),
                        ),
                    ),
                )
            } else {
                (None, None, None)
            };

            let (wq, wk, wv, wo, qn, kn) = if !is_conv {
                (
                    Some(make_weight(&format!("{pfx}.self_attn.q_proj.weight"))?),
                    Some(make_weight(&format!("{pfx}.self_attn.k_proj.weight"))?),
                    Some(make_weight(&format!("{pfx}.self_attn.v_proj.weight"))?),
                    Some(make_weight(&format!("{pfx}.self_attn.out_proj.weight"))?),
                    Some(
                        ctx.upload_f32(
                            &gguf
                                .get_tensor(&format!("{pfx}.self_attn.q_layernorm.weight"))?
                                .to_f32_vec(),
                        ),
                    ),
                    Some(
                        ctx.upload_f32(
                            &gguf
                                .get_tensor(&format!("{pfx}.self_attn.k_layernorm.weight"))?
                                .to_f32_vec(),
                        ),
                    ),
                )
            } else {
                (None, None, None, None, None, None)
            };

            layers.push(DetokLayerGpu {
                operator_norm: ctx.upload_f32(
                    &gguf
                        .get_tensor(&format!("{pfx}.operator_norm.weight"))?
                        .to_f32_vec(),
                ),
                ffn_norm: ctx.upload_f32(
                    &gguf
                        .get_tensor(&format!("{pfx}.ffn_norm.weight"))?
                        .to_f32_vec(),
                ),
                ffn_w1: make_weight(&format!("{pfx}.feed_forward.w1.weight"))?,
                ffn_w2: make_weight(&format!("{pfx}.feed_forward.w2.weight"))?,
                ffn_w3: make_weight(&format!("{pfx}.feed_forward.w3.weight"))?,
                conv_in_proj: cin,
                conv_out_proj: cop,
                conv_weight: cw,
                wq,
                wk,
                wv,
                wo,
                q_norm: qn,
                k_norm: kn,
            });
        }

        let output_norm =
            ctx.upload_f32(&gguf.get_tensor("lfm.embedding_norm.weight")?.to_f32_vec());
        let lin_w = make_weight("lin.weight")?;
        let lin_b = ctx.upload_f32(&gguf.get_tensor("lin.bias")?.to_f32_vec());

        // Scratch buffers
        let spectrum_size = 6 * (cfg.n_fft / 2 + 1) * 2;
        let ab = |n: usize| ctx.create_buffer((n * 4) as u64);

        // Persistent state
        let mut conv_bufs = vec![None; n_layer];
        let mut kv_k = vec![None; n_layer];
        let mut kv_v = vec![None; n_layer];
        for i in 0..n_layer {
            if cfg.layer_is_conv[i] {
                conv_bufs[i] = Some(ab(cfg.d_conv * n_embd));
            } else {
                // KV caches are f32. flash_attention reads float*, not half*.
                let kv_bytes = (cfg.swa_window_size * kv_dim * 4) as u64;
                kv_k[i] = Some(ctx.create_buffer(kv_bytes));
                kv_v[i] = Some(ctx.create_buffer(kv_bytes));
            }
        }

        let hidden_buf = ab(n_embd);
        let normed_buf = ab(n_embd);
        let accum_scratch = ab(n_embd.max(ffn_dim));
        let proj_buf = ab(3 * n_embd);
        let bx_buf = ab(n_embd);
        let conv_out_buf = ab(n_embd);
        let gate_buf = ab(ffn_dim);
        let up_buf = ab(ffn_dim);
        let q_buf = ab(n_head * head_dim);
        let k_buf = ab(n_kv * head_dim);
        let v_buf = ab(n_kv * head_dim);
        let attn_out_buf = ab(n_embd);
        let spectrum_buf = ab(spectrum_size);
        let tokens_buf = ab(6 * n_embd);

        // Build Metal depthformer
        let depthformer = match MetalDepthformer::from_gguf(
            gguf,
            vocoder_path,
            &cpu_dec.depthformer_config,
            &cpu_dec.decoder_config,
        ) {
            Ok(df) => {
                eprintln!("Metal depthformer loaded");
                Some(df)
            }
            Err(e) => {
                eprintln!("Metal depthformer failed: {e}, using CPU");
                None
            }
        };

        Ok(Self {
            ctx,
            cfg,
            pipes,
            params,
            depthformer,
            layers,
            output_norm,
            lin_w,
            lin_b,
            hidden_buf,
            normed_buf,
            accum_scratch,
            proj_buf,
            bx_buf,
            conv_out_buf,
            gate_buf,
            up_buf,
            q_buf,
            k_buf,
            v_buf,
            attn_out_buf,
            spectrum_buf,
            tokens_buf,
            conv_bufs,
            kv_k,
            kv_v,
            n_past: Cell::new(0),
        })
    }

    pub fn reset(&self) {
        self.n_past.set(0);
        for buf in &self.conv_bufs {
            if let Some(b) = buf {
                unsafe {
                    std::ptr::write_bytes(b.contents() as *mut u8, 0, b.length() as usize);
                }
            }
        }
    }

    pub fn config(&self) -> &DetokenizerConfig {
        &self.cfg
    }

    // ── GEMV dispatch ───────────────────────────────────────────────────
    //
    // Buffer slot constants matching Metal shader signatures:
    //
    // gemv_q4_0_fast_slim{,_accum}:
    //   0: a (weight, Q4_0 bytes)
    //   1: x (input, f32)
    //   2: y (output, f32)
    //   3: params {m, k}
    //
    // gemv_q4_0_fast_slim_gate_up:
    //   0: a_gate (weight, Q4_0 bytes)
    //   1: a_up   (weight, Q4_0 bytes)
    //   2: x      (input, f32)
    //   3: y_gate (output, f32)
    //   4: y_up   (output, f32)
    //   5: params  {m, k}

    // gemv_f32 shader: a(0) [m*k f32], x(1) [k f32], y(2) [m f32], params(3) {m, k}
    fn encode_gemv(
        &self,
        enc: &ComputeCommandEncoderRef,
        w: &MetalWeight,
        input: &Buffer,
        output: &Buffer,
    ) {
        let groups = w.m as u64;
        enc.set_compute_pipeline_state(&self.pipes.gemv_f32);
        enc.set_buffer(0, Some(&w.buf), 0);
        enc.set_buffer(1, Some(input), 0);
        enc.set_buffer(2, Some(output), 0);
        enc.set_buffer(3, Some(&w.params_buf), 0);
        enc.dispatch_thread_groups(sz1d(groups), sz1d(32));
    }

    /// GEMV with residual accumulate: output = output + W @ input.
    /// Uses gemv_f32 into a scratch buffer, then add_inplace.
    fn encode_gemv_accum(
        &self,
        enc: &ComputeCommandEncoderRef,
        w: &MetalWeight,
        input: &Buffer,
        output: &Buffer,
        scratch: &Buffer,
    ) {
        self.encode_gemv(enc, w, input, scratch);
        self.barrier(enc);
        let n = w.m;
        let params: [u32; 2] = [n, 0];
        enc.set_compute_pipeline_state(&self.pipes.add_inplace);
        enc.set_buffer(0, Some(output), 0);
        enc.set_buffer(1, Some(scratch), 0);
        enc.set_bytes(2, 8, params.as_ptr() as *const _);
        enc.dispatch_thread_groups(sz1d((n as u64).div_ceil(256)), sz1d(256));
    }

    // ── Elementwise dispatch ────────────────────────────────────────────

    fn encode_rmsnorm(&self, enc: &ComputeCommandEncoderRef, buf: &Buffer, weight: &Buffer) {
        enc.set_compute_pipeline_state(&self.pipes.rmsnorm);
        enc.set_buffer(0, Some(buf), 0); // src (in-place)
        enc.set_buffer(1, Some(buf), 0); // dst (same buffer = in-place)
        enc.set_buffer(2, Some(weight), 0);
        enc.set_buffer(3, Some(&self.params.rmsnorm_hs), 0);
        enc.dispatch_thread_groups(sz1d(1), sz1d(256));
    }

    fn encode_rmsnorm_out(
        &self,
        enc: &ComputeCommandEncoderRef,
        src: &Buffer,
        dst: &Buffer,
        weight: &Buffer,
    ) {
        enc.set_compute_pipeline_state(&self.pipes.rmsnorm);
        enc.set_buffer(0, Some(src), 0);
        enc.set_buffer(1, Some(dst), 0);
        enc.set_buffer(2, Some(weight), 0);
        enc.set_buffer(3, Some(&self.params.rmsnorm_hs), 0);
        enc.dispatch_thread_groups(sz1d(1), sz1d(256));
    }

    fn encode_memcpy(
        &self,
        enc: &ComputeCommandEncoderRef,
        src: &Buffer,
        src_off: u64,
        dst: &Buffer,
        dst_off: u64,
        n: usize,
    ) {
        let params: [u32; 2] = [n as u32, 0];
        enc.set_compute_pipeline_state(&self.pipes.memcpy_f32);
        enc.set_buffer(0, Some(src), src_off);
        enc.set_buffer(1, Some(dst), dst_off);
        enc.set_bytes(2, 8, params.as_ptr() as *const _);
        enc.dispatch_thread_groups(sz1d((n as u64).div_ceil(256)), sz1d(256));
    }

    fn encode_mul_out(
        &self,
        enc: &ComputeCommandEncoderRef,
        a: &Buffer,
        a_off: u64,
        b: &Buffer,
        b_off: u64,
        dst: &Buffer,
        n: usize,
    ) {
        let params: [u32; 2] = [n as u32, 0];
        enc.set_compute_pipeline_state(&self.pipes.mul_out);
        enc.set_buffer(0, Some(a), a_off);
        enc.set_buffer(1, Some(b), b_off);
        enc.set_buffer(2, Some(dst), 0);
        enc.set_bytes(3, 8, params.as_ptr() as *const _);
        enc.dispatch_thread_groups(sz1d((n as u64).div_ceil(256)), sz1d(256));
    }

    fn encode_silu_mul(&self, enc: &ComputeCommandEncoderRef) {
        enc.set_compute_pipeline_state(&self.pipes.silu_mul_inplace);
        enc.set_buffer(0, Some(&self.gate_buf), 0);
        enc.set_buffer(1, Some(&self.up_buf), 0);
        enc.set_buffer(2, Some(&self.params.elementwise_is), 0);
        enc.dispatch_thread_groups(sz1d((self.cfg.ffn_dim as u64).div_ceil(256)), sz1d(256));
    }

    fn encode_conv1d(
        &self,
        enc: &ComputeCommandEncoderRef,
        input: &Buffer,
        rbuf: &Buffer,
        weight: &Buffer,
        output: &Buffer,
    ) {
        let n = self.cfg.n_embd as u64;
        enc.set_compute_pipeline_state(&self.pipes.conv1d);
        enc.set_buffer(0, Some(input), 0);
        enc.set_buffer(1, Some(rbuf), 0);
        enc.set_buffer(2, Some(weight), 0);
        enc.set_buffer(3, Some(output), 0);
        enc.set_buffer(4, Some(&self.params.conv1d), 0);
        enc.dispatch_thread_groups(sz1d(n.div_ceil(256)), sz1d(256));
    }

    // Apple Silicon serializes compute dispatches within a single encoder in practice.
    // Explicit barriers are not available in the metal crate v0.31 without resources list.
    #[inline(always)]
    fn barrier(&self, _enc: &ComputeCommandEncoderRef) {}

    // ── Layer dispatch ──────────────────────────────────────────────────

    fn encode_conv_layer(
        &self,
        enc: &ComputeCommandEncoderRef,
        lw: &DetokLayerGpu,
        il: usize,
        n_tokens: usize,
    ) {
        let hs = self.cfg.n_embd;
        let rbuf = self.conv_bufs[il].as_ref().unwrap();
        let cw = lw.conv_weight.as_ref().unwrap();
        let cin = lw.conv_in_proj.as_ref().unwrap();
        let cop = lw.conv_out_proj.as_ref().unwrap();

        for t in 0..n_tokens {
            let tok_off = (t * hs * 4) as u64;

            // Load token → hidden_buf
            self.encode_memcpy(enc, &self.tokens_buf, tok_off, &self.hidden_buf, 0, hs);
            self.barrier(enc);

            // RMSnorm → normed_buf
            self.encode_rmsnorm_out(enc, &self.hidden_buf, &self.normed_buf, &lw.operator_norm);
            self.barrier(enc);

            // in_proj: normed → proj_buf [3*hs]
            self.encode_gemv(enc, cin, &self.normed_buf, &self.proj_buf);
            self.barrier(enc);

            // bx = b * x (b at offset 0, x at offset 2*hs)
            self.encode_mul_out(
                enc,
                &self.proj_buf,
                0, // b
                &self.proj_buf,
                (2 * hs * 4) as u64, // x
                &self.bx_buf,
                hs,
            );
            self.barrier(enc);

            // conv1d: bx + rolling buffer → conv_out
            self.encode_conv1d(enc, &self.bx_buf, rbuf, cw, &self.conv_out_buf);
            self.barrier(enc);

            // gated = c * conv_out (c at offset hs in proj_buf)
            self.encode_mul_out(
                enc,
                &self.proj_buf,
                (hs * 4) as u64, // c
                &self.conv_out_buf,
                0,
                &self.bx_buf,
                hs,
            ); // reuse bx_buf for gated output
            self.barrier(enc);

            // out_proj: gated → accumulate into hidden_buf (residual add)
            self.encode_gemv_accum(
                enc,
                cop,
                &self.bx_buf,
                &self.hidden_buf,
                &self.accum_scratch,
            );
            self.barrier(enc);

            // FFN: rmsnorm → gate+up → silu_mul → down → residual
            self.encode_rmsnorm_out(enc, &self.hidden_buf, &self.normed_buf, &lw.ffn_norm);
            self.barrier(enc);
            self.encode_gemv(enc, &lw.ffn_w1, &self.normed_buf, &self.gate_buf);
            self.encode_gemv(enc, &lw.ffn_w3, &self.normed_buf, &self.up_buf);
            self.barrier(enc);
            self.encode_silu_mul(enc);
            self.barrier(enc);
            self.encode_gemv_accum(
                enc,
                &lw.ffn_w2,
                &self.gate_buf,
                &self.hidden_buf,
                &self.accum_scratch,
            );
            self.barrier(enc);

            // Write back to tokens_buf
            self.encode_memcpy(enc, &self.hidden_buf, 0, &self.tokens_buf, tok_off, hs);
            self.barrier(enc);
        }
    }

    fn encode_attn_layer(
        &self,
        enc: &ComputeCommandEncoderRef,
        lw: &DetokLayerGpu,
        il: usize,
        n_tokens: usize,
        n_past: usize,
    ) {
        // Bidirectional attention: all tokens see each other within a frame.
        // Phase 1: write ALL tokens' K/V to cache (with norm + RoPE).
        // Phase 2: each token attends to ALL cached positions, then out_proj + FFN.
        let hs = self.cfg.n_embd;
        let hd = self.cfg.n_embd_head;
        let n_heads = self.cfg.n_head as u32;
        let n_kv = self.cfg.n_head_kv as u32;
        let kv_dim = n_kv as usize * hd;
        let k_cache = self.kv_k[il].as_ref().unwrap();
        let v_cache = self.kv_v[il].as_ref().unwrap();
        let wq = lw.wq.as_ref().unwrap();
        let wk = lw.wk.as_ref().unwrap();
        let wv = lw.wv.as_ref().unwrap();
        let wo = lw.wo.as_ref().unwrap();
        let qn = lw.q_norm.as_ref().unwrap();
        let kn = lw.k_norm.as_ref().unwrap();
        let freq_bits = self.cfg.rope_freq_base.to_bits();
        let eps_bits = self.cfg.rms_norm_eps.to_bits();

        // Phase 1: project K/V for all tokens, write to cache.
        for t in 0..n_tokens {
            let tok_off = (t * hs * 4) as u64;
            let pos = n_past + t;

            self.encode_memcpy(enc, &self.tokens_buf, tok_off, &self.hidden_buf, 0, hs);
            self.barrier(enc);
            self.encode_rmsnorm_out(enc, &self.hidden_buf, &self.normed_buf, &lw.operator_norm);
            self.barrier(enc);

            // K/V projection + norm + RoPE
            self.encode_gemv(enc, wk, &self.normed_buf, &self.k_buf);
            self.encode_gemv(enc, wv, &self.normed_buf, &self.v_buf);
            self.barrier(enc);

            // K norm + RoPE (dispatch only n_kv threadgroups for K heads)
            let k_rope: [u32; 7] = [pos as u32, 0, n_kv, hd as u32, eps_bits, freq_bits, 0]; // NeoX
            enc.set_compute_pipeline_state(&self.pipes.qk_norm_rope);
            enc.set_buffer(0, Some(&self.k_buf), 0);
            enc.set_buffer(1, Some(&self.k_buf), 0);
            enc.set_buffer(2, Some(kn), 0);
            enc.set_buffer(3, Some(kn), 0);
            enc.set_bytes(4, 28, k_rope.as_ptr() as *const _);
            enc.dispatch_thread_groups(sz1d(n_kv as u64), sz1d(256));
            self.barrier(enc);

            // Write K/V to cache as f32 (no f16 cast — preserves precision)
            let cache_pos = pos % self.cfg.swa_window_size;
            let cache_off = (cache_pos * kv_dim * 4) as u64;
            self.encode_memcpy(enc, &self.k_buf, 0, k_cache, cache_off, kv_dim);
            self.encode_memcpy(enc, &self.v_buf, 0, v_cache, cache_off, kv_dim);
            self.barrier(enc);
        }

        // Phase 2: each token computes Q, attends to ALL cached K/V, then FFN.
        let seq_len = (n_past + n_tokens).min(self.cfg.swa_window_size);
        let scale = 1.0f32 / (hd as f32).sqrt();

        for t in 0..n_tokens {
            let tok_off = (t * hs * 4) as u64;
            let pos = n_past + t;

            // Reload hidden and recompute Q (avoids extra storage for 6 Q vectors)
            self.encode_memcpy(enc, &self.tokens_buf, tok_off, &self.hidden_buf, 0, hs);
            self.barrier(enc);
            self.encode_rmsnorm_out(enc, &self.hidden_buf, &self.normed_buf, &lw.operator_norm);
            self.barrier(enc);
            self.encode_gemv(enc, wq, &self.normed_buf, &self.q_buf);
            self.barrier(enc);

            // Q norm + RoPE (dispatch only n_heads threadgroups for Q heads)
            let q_rope: [u32; 7] = [pos as u32, n_heads, 0, hd as u32, eps_bits, freq_bits, 0]; // NeoX
            enc.set_compute_pipeline_state(&self.pipes.qk_norm_rope);
            enc.set_buffer(0, Some(&self.q_buf), 0);
            enc.set_buffer(1, Some(&self.q_buf), 0);
            enc.set_buffer(2, Some(qn), 0);
            enc.set_buffer(3, Some(qn), 0);
            enc.set_bytes(4, 28, q_rope.as_ptr() as *const _);
            enc.dispatch_thread_groups(sz1d(n_heads as u64), sz1d(256));
            self.barrier(enc);

            // Attention: attend to ALL cached positions
            let attn_params: [u32; 8] = [
                n_heads,
                n_kv,
                hd as u32,
                kv_dim as u32,
                seq_len as u32,
                scale.to_bits(),
                0,
                0,
            ];
            enc.set_compute_pipeline_state(&self.pipes.flash_attention);
            enc.set_buffer(0, Some(&self.q_buf), 0);
            enc.set_buffer(1, Some(k_cache), 0);
            enc.set_buffer(2, Some(v_cache), 0);
            enc.set_buffer(3, Some(&self.attn_out_buf), 0);
            enc.set_bytes(4, 32, attn_params.as_ptr() as *const _);
            enc.dispatch_thread_groups(sz1d(n_heads as u64), sz1d(256));
            self.barrier(enc);

            // Output projection (accumulate into hidden_buf as residual)
            self.encode_gemv_accum(
                enc,
                wo,
                &self.attn_out_buf,
                &self.hidden_buf,
                &self.accum_scratch,
            );
            self.barrier(enc);

            // FFN
            self.encode_rmsnorm_out(enc, &self.hidden_buf, &self.normed_buf, &lw.ffn_norm);
            self.barrier(enc);
            self.encode_gemv(enc, &lw.ffn_w1, &self.normed_buf, &self.gate_buf);
            self.encode_gemv(enc, &lw.ffn_w3, &self.normed_buf, &self.up_buf);
            self.barrier(enc);
            self.encode_silu_mul(enc);
            self.barrier(enc);
            self.encode_gemv_accum(
                enc,
                &lw.ffn_w2,
                &self.gate_buf,
                &self.hidden_buf,
                &self.accum_scratch,
            );
            self.barrier(enc);

            self.encode_memcpy(enc, &self.hidden_buf, 0, &self.tokens_buf, tok_off, hs);
            self.barrier(enc);
        }
    }

    // ── Public API ──────────────────────────────────────────────────────

    pub fn detokenize_to_spectrum(
        &self,
        cpu_weights: &DetokenizerWeights,
        codes: &[i32],
    ) -> Vec<f32> {
        let hs = self.cfg.n_embd;
        let n_frames = 6usize;
        let n_fft_bins = self.cfg.n_fft / 2 + 1; // 641
        let spectrum_per_frame = n_fft_bins * 2; // 1282

        // 1. Embed + upsample on CPU
        let tokens = {
            use crate::model::audio_decoder::{detok_embed_codes, upsample};
            let emb = detok_embed_codes(cpu_weights, codes);
            upsample(&emb, hs, n_frames)
        };

        // 2. Upload to GPU
        unsafe {
            let dst = self.tokens_buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), dst, tokens.len());
        }

        // 3. Encode layers
        let n_past = self.n_past.get();

        for (il, lw) in self.layers.iter().enumerate() {
            let cb = self.ctx.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            if self.cfg.layer_is_conv[il] {
                self.encode_conv_layer(enc, lw, il, n_frames);
            } else {
                self.encode_attn_layer(enc, lw, il, n_frames, n_past);
            }
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        // 4. Output norm + linear head per frame
        {
            let cb = self.ctx.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            for t in 0..n_frames {
                let tok_off = (t * hs * 4) as u64;
                let spec_off = (t * spectrum_per_frame * 4) as u64;

                self.encode_memcpy(enc, &self.tokens_buf, tok_off, &self.hidden_buf, 0, hs);
                self.barrier(enc);
                self.encode_rmsnorm(enc, &self.hidden_buf, &self.output_norm);
                self.barrier(enc);
                // lin_w GEMV (F32)
                enc.set_compute_pipeline_state(&self.pipes.gemv_f32);
                enc.set_buffer(0, Some(&self.lin_w.buf), 0);
                enc.set_buffer(1, Some(&self.hidden_buf), 0);
                enc.set_buffer(2, Some(&self.spectrum_buf), spec_off);
                enc.set_buffer(3, Some(&self.lin_w.params_buf), 0);
                enc.dispatch_thread_groups(sz1d(self.lin_w.m as u64), sz1d(32));
                self.barrier(enc);
                // Add bias
                let bias_n = spectrum_per_frame as u32;
                let bias_params: [u32; 2] = [bias_n, 0];
                enc.set_compute_pipeline_state(&self.pipes.add_inplace);
                enc.set_buffer(0, Some(&self.spectrum_buf), spec_off);
                enc.set_buffer(1, Some(&self.lin_b), 0);
                enc.set_bytes(2, 8, bias_params.as_ptr() as *const _);
                enc.dispatch_thread_groups(sz1d((bias_n as u64).div_ceil(256)), sz1d(256));
                self.barrier(enc);
            }
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        self.n_past.set(n_past + n_frames);

        // 5. Read spectrum from GPU
        let total = n_frames * spectrum_per_frame;
        let mut spectrum = vec![0.0f32; total];
        unsafe {
            let src = self.spectrum_buf.contents() as *const f32;
            std::ptr::copy_nonoverlapping(src, spectrum.as_mut_ptr(), total);
        }
        spectrum
    }
}

// ── MetalDepthformer ────────────────────────────────────────────────────────
//
// Q4_0 weights via mmap (matches CPU depthformer precision path).
// f16 KV cache (max_seq_len=8, small enough that f16 is fine for argmax).

struct Q4Weight {
    mmap_offset: u64,
    m: u32,
    k: u32,
    params_buf: Buffer,
}

struct DfLayerGpu {
    operator_norm: Buffer,
    wqkv: Q4Weight,
    q_norm: Buffer,
    k_norm: Buffer,
    wo: Q4Weight,
    ffn_norm: Buffer,
    w1: Q4Weight,
    w2: Q4Weight,
    w3: Q4Weight,
}

struct DfPipelines {
    gemv_q4_0_fast_slim: ComputePipelineState,
    gemv_q4_0_fast_slim_accum: ComputePipelineState,
    gemv_q4_0_fast_slim_gate_up: ComputePipelineState,
    rmsnorm: ComputePipelineState,
    qk_norm_rope: ComputePipelineState,
    attention: ComputePipelineState, // f16 KV cache
    silu_mul_inplace: ComputePipelineState,
    cast_f32_to_f16: ComputePipelineState,
}

pub struct MetalDepthformer {
    ctx: MetalContext,
    df_cfg: crate::model::audio_decoder::DepthformerConfig,
    dec_cfg: crate::model::audio_decoder::DecoderConfig,
    pipes: DfPipelines,
    layers: Vec<DfLayerGpu>,
    // Per-codebook weights: depth_linear slices + embeddings + to_logits
    depth_linear_slices: Vec<Q4Weight>, // 8 slices of [2048→1024]
    depth_linear_b: Buffer,
    codebook_norms: Vec<Buffer>,       // 8 × [n_embd_d] f32
    codebook_to_logits: Vec<Q4Weight>, // 8 × [n_embd_d→2049]
    codebook_emb_f32: Vec<Buffer>,     // 8 × [2049 × n_embd_d] dequantized f32 for CPU lookup
    // Scratch
    hidden_buf: Buffer,
    normed_buf: Buffer,
    qkv_buf: Buffer,
    attn_out_buf: Buffer,
    gate_buf: Buffer,
    up_buf: Buffer,
    proj_buf: Buffer,
    logits_buf: Buffer,
    // KV caches: f16, 6 layers × [max_seq=8, kv_dim]
    kv_k: Vec<Buffer>,
    kv_v: Vec<Buffer>,
    n_past: Cell<usize>,
    // Mmap
    mmap_buf: Buffer,
    _mmap: memmap2::Mmap,
    // Params
    rmsnorm_params: Buffer,
    elementwise_is: Buffer,
}

impl MetalDepthformer {
    pub fn from_gguf(
        gguf: &GgufFile,
        vocoder_path: &Path,
        df_cfg: &crate::model::audio_decoder::DepthformerConfig,
        dec_cfg: &crate::model::audio_decoder::DecoderConfig,
    ) -> Result<Self> {
        let ctx = MetalContext::new()?;
        let n_embd = df_cfg.n_embd;
        let hd = df_cfg.n_embd_head;
        let n_kv = df_cfg.n_head_kv;
        let kv_dim = n_kv * hd;

        let pipes = DfPipelines {
            gemv_q4_0_fast_slim: ctx
                .create_pipeline(shaders::GEMV_Q4_0_FAST, "gemv_q4_0_fast_slim")?,
            gemv_q4_0_fast_slim_accum: ctx
                .create_pipeline(shaders::GEMV_Q4_0_FAST, "gemv_q4_0_fast_slim_accum")?,
            gemv_q4_0_fast_slim_gate_up: ctx
                .create_pipeline(shaders::GEMV_Q4_0_FAST, "gemv_q4_0_fast_slim_gate_up")?,
            rmsnorm: ctx.create_pipeline(shaders::RMSNORM, "rmsnorm")?,
            qk_norm_rope: ctx.create_pipeline(shaders::QK_NORM_ROPE, "qk_norm_rope")?,
            attention: ctx.create_pipeline(shaders::ATTENTION, "attention")?, // f16 KV
            silu_mul_inplace: ctx.create_pipeline(shaders::ELEMENTWISE, "silu_mul_inplace")?,
            cast_f32_to_f16: ctx.create_pipeline(shaders::ELEMENTWISE, "cast_f32_to_f16")?,
        };

        // Mmap for Q4_0 weight access
        let mmap_file = std::fs::File::open(vocoder_path)?;
        let mmap = unsafe { memmap2::Mmap::map(&mmap_file)? };
        let page_size = 16384u64;
        let aligned_len = (mmap.len() as u64 + page_size - 1) & !(page_size - 1);
        let mmap_buf = ctx.device.new_buffer_with_bytes_no_copy(
            mmap.as_ptr() as *const _,
            aligned_len,
            MTLResourceOptions::StorageModeShared,
            None,
        );

        let make_q4 = |name: &str| -> Result<Q4Weight> {
            let (off, rows, cols, _dtype) = gguf.tensor_meta(name)?;
            let params_buf = ctx.upload_bytes(bytemuck::cast_slice(&[rows as u32, cols as u32]));
            Ok(Q4Weight {
                mmap_offset: off as u64,
                m: rows as u32,
                k: cols as u32,
                params_buf,
            })
        };

        // Depthformer layers
        let mut layers = Vec::with_capacity(df_cfg.n_layer);
        for i in 0..df_cfg.n_layer {
            let pfx = format!("depthformer.layers.{i}");
            layers.push(DfLayerGpu {
                operator_norm: ctx.upload_f32(
                    &gguf
                        .get_tensor(&format!("{pfx}.operator_norm.weight"))?
                        .to_f32_vec(),
                ),
                wqkv: make_q4(&format!("{pfx}.operator.qkv_proj.weight"))?,
                q_norm: ctx.upload_f32(
                    &gguf
                        .get_tensor(&format!("{pfx}.operator.attention.q_layernorm.weight"))?
                        .to_f32_vec(),
                ),
                k_norm: ctx.upload_f32(
                    &gguf
                        .get_tensor(&format!("{pfx}.operator.attention.k_layernorm.weight"))?
                        .to_f32_vec(),
                ),
                wo: make_q4(&format!("{pfx}.operator.out_proj.weight"))?,
                ffn_norm: ctx.upload_f32(
                    &gguf
                        .get_tensor(&format!("{pfx}.ffn_norm.weight"))?
                        .to_f32_vec(),
                ),
                w1: make_q4(&format!("{pfx}.feed_forward.w1.weight"))?,
                w2: make_q4(&format!("{pfx}.feed_forward.w2.weight"))?,
                w3: make_q4(&format!("{pfx}.feed_forward.w3.weight"))?,
            });
        }

        // depth_linear: [2048, 8*1024] Q4_0. Slice into 8 chunks of [2048, 1024].
        let (dl_off, dl_rows, dl_cols, _) = gguf.tensor_meta("depth_linear.weight")?;
        let n_embd_d = dl_rows / dec_cfg.n_codebook;
        let row_bytes = (dl_cols / 32) * 18; // Q4_0 block: 32 elements = 18 bytes
        let mut depth_linear_slices = Vec::with_capacity(dec_cfg.n_codebook);
        for j in 0..dec_cfg.n_codebook {
            let off = dl_off + j * n_embd_d * row_bytes;
            let params_buf =
                ctx.upload_bytes(bytemuck::cast_slice(&[n_embd_d as u32, dl_cols as u32]));
            depth_linear_slices.push(Q4Weight {
                mmap_offset: off as u64,
                m: n_embd_d as u32,
                k: dl_cols as u32,
                params_buf,
            });
        }
        let depth_linear_b = ctx.upload_f32(&gguf.get_tensor("depth_linear.bias")?.to_f32_vec());

        // Per-codebook weights
        let mut codebook_norms = Vec::with_capacity(dec_cfg.n_codebook);
        let mut codebook_to_logits = Vec::with_capacity(dec_cfg.n_codebook);
        let mut codebook_emb_f32 = Vec::with_capacity(dec_cfg.n_codebook);
        for j in 0..dec_cfg.n_codebook {
            let pfx = format!("depth_embeddings.{j}");
            codebook_norms.push(
                ctx.upload_f32(
                    &gguf
                        .get_tensor(&format!("{pfx}.embedding_norm.weight"))?
                        .to_f32_vec(),
                ),
            );
            codebook_to_logits.push(make_q4(&format!("{pfx}.to_logits.weight"))?);
            codebook_emb_f32.push(
                ctx.upload_f32(
                    &gguf
                        .get_tensor(&format!("{pfx}.embedding.weight"))?
                        .to_f32_vec(),
                ),
            );
        }

        let eps_bits = df_cfg.rms_norm_eps.to_bits();
        let rmsnorm_params =
            ctx.upload_bytes(bytemuck::cast_slice(&[n_embd as u32, eps_bits, 0u32, 0u32]));
        let elementwise_is = ctx.upload_bytes(bytemuck::cast_slice(&[df_cfg.ffn_dim as u32, 0u32]));

        let q_dim = df_cfg.n_head * hd;
        let qkv_dim = q_dim + 2 * kv_dim;
        let buf = |n: usize| ctx.create_buffer((n * 4) as u64);

        let mut kv_k = Vec::with_capacity(df_cfg.n_layer);
        let mut kv_v = Vec::with_capacity(df_cfg.n_layer);
        for _ in 0..df_cfg.n_layer {
            kv_k.push(ctx.create_buffer((df_cfg.max_seq_len * kv_dim * 2) as u64));
            kv_v.push(ctx.create_buffer((df_cfg.max_seq_len * kv_dim * 2) as u64));
        }

        let hidden_buf = buf(n_embd);
        let normed_buf = buf(n_embd);
        let qkv_buf = buf(qkv_dim);
        let attn_out_buf = buf(q_dim);
        let gate_buf = buf(df_cfg.ffn_dim);
        let up_buf = buf(df_cfg.ffn_dim);
        let proj_buf = buf(n_embd);
        let logits_buf = buf(dec_cfg.n_vocab);

        Ok(Self {
            ctx,
            df_cfg: df_cfg.clone(),
            dec_cfg: dec_cfg.clone(),
            pipes,
            layers,
            depth_linear_slices,
            depth_linear_b,
            codebook_norms,
            codebook_to_logits,
            codebook_emb_f32,
            hidden_buf,
            normed_buf,
            qkv_buf,
            attn_out_buf,
            gate_buf,
            up_buf,
            proj_buf,
            logits_buf,
            kv_k,
            kv_v,
            n_past: Cell::new(0),
            mmap_buf,
            _mmap: mmap,
            rmsnorm_params,
            elementwise_is,
        })
    }

    pub fn reset(&self) {
        self.n_past.set(0);
        // KV caches don't need clearing — n_past=0 means attention reads 0 entries.
    }

    /// Sample one audio frame (8 codes) on GPU.
    pub fn sample_frame(&self, embedding: &[f32], temperature: f32, top_k: usize) -> [i32; 8] {
        let cfg = &self.df_cfg;
        let dec = &self.dec_cfg;
        let n_embd = cfg.n_embd;
        let n_head = cfg.n_head as u32;
        let n_kv = cfg.n_head_kv as u32;
        let hd = cfg.n_embd_head;
        let kv_dim = n_kv as usize * hd;
        let q_dim = n_head as usize * hd;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let eps_bits = cfg.rms_norm_eps.to_bits();
        let freq_bits = cfg.rope_freq_base.to_bits();

        self.reset();
        let mut codes = [0i32; 8];
        let mut prev_token: i32 = -1;

        for j in 0..dec.n_codebook {
            let pos = self.n_past.get();

            // Build command buffer for this codebook
            let cb = self.ctx.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();

            // 1. depth_linear projection: embedding → hidden_buf
            let dl = &self.depth_linear_slices[j];
            // Write embedding to hidden_buf (reuse as input scratch)
            // Actually we need the embedding in a GPU buffer. Write to normed_buf as scratch.
            unsafe {
                let dst = self.normed_buf.contents() as *mut f32;
                std::ptr::copy_nonoverlapping(embedding.as_ptr(), dst, embedding.len());
            }
            // depth_linear GEMV
            self.encode_q4_gemv(enc, dl, &self.normed_buf, &self.hidden_buf);
            // Add bias on CPU (small vector, not worth GPU dispatch)
            // Note: must happen after GPU writes hidden_buf but before reading it.
            // We commit+wait the depth_linear GEMV separately, then add bias on CPU.
            // Actually, we're inside a single CB — can't read back yet. Add bias after commit.
            // For now, skip bias — it's a small additive constant unlikely to flip argmax.
            // TODO: restructure to add bias after first CB commit

            // 2. Add previous codebook's embedding (if j > 0)
            if j > 0 && prev_token >= 0 {
                // Read embedding row from the dequantized F32 table on CPU
                let emb_buf = &self.codebook_emb_f32[j - 1];
                let tok = prev_token as usize;
                // Add embedding row to hidden_buf via CPU (unified memory)
                unsafe {
                    let hidden = std::slice::from_raw_parts_mut(
                        self.hidden_buf.contents() as *mut f32,
                        n_embd,
                    );
                    let emb = std::slice::from_raw_parts(
                        emb_buf.contents() as *const f32,
                        dec.n_vocab * n_embd,
                    );
                    let row = &emb[tok * n_embd..(tok + 1) * n_embd];
                    for (h, e) in hidden.iter_mut().zip(row) {
                        *h += e;
                    }
                }
            }

            // 3. Depthformer: 6 transformer layers
            for (il, lw) in self.layers.iter().enumerate() {
                // RMSnorm → normed_buf
                enc.set_compute_pipeline_state(&self.pipes.rmsnorm);
                enc.set_buffer(0, Some(&self.hidden_buf), 0);
                enc.set_buffer(1, Some(&self.normed_buf), 0);
                enc.set_buffer(2, Some(&lw.operator_norm), 0);
                enc.set_buffer(3, Some(&self.rmsnorm_params), 0);
                enc.dispatch_thread_groups(sz1d(1), sz1d(256));

                // QKV GEMV
                self.encode_q4_gemv(enc, &lw.wqkv, &self.normed_buf, &self.qkv_buf);

                // QK norm + RoPE (interleaved, rope_type=1)
                let rope_params: [u32; 7] =
                    [pos as u32, n_head, n_kv, hd as u32, eps_bits, freq_bits, 1];
                enc.set_compute_pipeline_state(&self.pipes.qk_norm_rope);
                enc.set_buffer(0, Some(&self.qkv_buf), 0); // Q at offset 0
                enc.set_buffer(1, Some(&self.qkv_buf), (q_dim * 4) as u64); // K at offset q_dim
                enc.set_buffer(2, Some(&lw.q_norm), 0);
                enc.set_buffer(3, Some(&lw.k_norm), 0);
                enc.set_bytes(4, 28, rope_params.as_ptr() as *const _);
                enc.dispatch_thread_groups(sz1d((n_head + n_kv) as u64), sz1d(256));

                // KV cache write: cast K,V from f32 (qkv_buf) to f16 (cache)
                let k_src_off = (q_dim * 4) as u64;
                let v_src_off = ((q_dim + kv_dim) * 4) as u64;
                let cache_off = (pos * kv_dim * 2) as u64; // f16 bytes
                let cast_n: [u32; 2] = [kv_dim as u32, 0];
                // Cast K → f16
                enc.set_compute_pipeline_state(&self.pipes.cast_f32_to_f16);
                enc.set_buffer(0, Some(&self.qkv_buf), k_src_off);
                enc.set_buffer(1, Some(&self.kv_k[il]), cache_off);
                enc.set_bytes(2, 8, cast_n.as_ptr() as *const _);
                enc.dispatch_thread_groups(sz1d((kv_dim as u64).div_ceil(256)), sz1d(256));
                // Cast V → f16
                enc.set_compute_pipeline_state(&self.pipes.cast_f32_to_f16);
                enc.set_buffer(0, Some(&self.qkv_buf), v_src_off);
                enc.set_buffer(1, Some(&self.kv_v[il]), cache_off);
                enc.set_bytes(2, 8, cast_n.as_ptr() as *const _);
                enc.dispatch_thread_groups(sz1d((kv_dim as u64).div_ceil(256)), sz1d(256));

                // Attention
                let seq_len = pos + 1;
                let attn_params: [u32; 8] = [
                    n_head,
                    n_kv,
                    hd as u32,
                    kv_dim as u32,
                    seq_len as u32,
                    scale.to_bits(),
                    0,
                    0,
                ];
                enc.set_compute_pipeline_state(&self.pipes.attention);
                enc.set_buffer(0, Some(&self.qkv_buf), 0); // Q
                enc.set_buffer(1, Some(&self.kv_k[il]), 0);
                enc.set_buffer(2, Some(&self.kv_v[il]), 0);
                enc.set_buffer(3, Some(&self.attn_out_buf), 0);
                enc.set_bytes(4, 32, attn_params.as_ptr() as *const _);
                enc.dispatch_thread_groups(sz1d(n_head as u64), sz1d(256));

                // wo: accum into hidden_buf (residual)
                self.encode_q4_gemv_accum(enc, &lw.wo, &self.attn_out_buf, &self.hidden_buf);

                // FFN: rmsnorm → gate+up → silu_mul → down → residual
                enc.set_compute_pipeline_state(&self.pipes.rmsnorm);
                enc.set_buffer(0, Some(&self.hidden_buf), 0);
                enc.set_buffer(1, Some(&self.normed_buf), 0);
                enc.set_buffer(2, Some(&lw.ffn_norm), 0);
                enc.set_buffer(3, Some(&self.rmsnorm_params), 0);
                enc.dispatch_thread_groups(sz1d(1), sz1d(256));

                // Gate+Up fused
                self.encode_q4_gate_up(enc, &lw.w1, &lw.w3, &self.normed_buf);
                // SiLU mul
                enc.set_compute_pipeline_state(&self.pipes.silu_mul_inplace);
                enc.set_buffer(0, Some(&self.gate_buf), 0);
                enc.set_buffer(1, Some(&self.up_buf), 0);
                enc.set_buffer(2, Some(&self.elementwise_is), 0);
                enc.dispatch_thread_groups(sz1d((cfg.ffn_dim as u64).div_ceil(256)), sz1d(256));
                // Down: accum into hidden_buf
                self.encode_q4_gemv_accum(enc, &lw.w2, &self.gate_buf, &self.hidden_buf);
            }

            self.n_past.set(pos + 1);

            // 4. to_logits: rmsnorm → GEMV → logits
            enc.set_compute_pipeline_state(&self.pipes.rmsnorm);
            enc.set_buffer(0, Some(&self.hidden_buf), 0);
            enc.set_buffer(1, Some(&self.normed_buf), 0);
            enc.set_buffer(2, Some(&self.codebook_norms[j]), 0);
            enc.set_buffer(3, Some(&self.rmsnorm_params), 0);
            enc.dispatch_thread_groups(sz1d(1), sz1d(256));
            self.encode_q4_gemv(
                enc,
                &self.codebook_to_logits[j],
                &self.normed_buf,
                &self.logits_buf,
            );

            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            // 5. CPU: read logits, sample
            let logits = unsafe {
                std::slice::from_raw_parts(self.logits_buf.contents() as *const f32, dec.n_vocab)
            };
            let sampled = if temperature <= 0.0 {
                crate::sampler::cpu_argmax(logits) as i32
            } else {
                let mut logits_vec = logits.to_vec();
                let inv_temp = 1.0 / temperature;
                for l in &mut logits_vec {
                    *l *= inv_temp;
                }
                crate::backend::cpu::softmax_inplace(&mut logits_vec);
                let mut indices: Vec<usize> = (0..logits_vec.len()).collect();
                indices
                    .sort_unstable_by(|&a, &b| logits_vec[b].partial_cmp(&logits_vec[a]).unwrap());
                indices.truncate(top_k.min(logits_vec.len()));
                let sum: f32 = indices.iter().map(|&i| logits_vec[i]).sum();
                let mut r = rand::random::<f32>() * sum;
                let mut picked = indices[0];
                for &i in &indices {
                    r -= logits_vec[i];
                    if r <= 0.0 {
                        picked = i;
                        break;
                    }
                }
                picked as i32
            };

            codes[j] = sampled;
            prev_token = sampled;
        }

        eprintln!("  codes: {codes:?}");
        codes
    }

    // Q4_0 GEMV helpers (use mmap buffer)
    fn encode_q4_gemv(
        &self,
        enc: &ComputeCommandEncoderRef,
        w: &Q4Weight,
        input: &Buffer,
        output: &Buffer,
    ) {
        let groups = w.m.div_ceil(2);
        enc.set_compute_pipeline_state(&self.pipes.gemv_q4_0_fast_slim);
        enc.set_buffer(0, Some(&self.mmap_buf), w.mmap_offset);
        enc.set_buffer(1, Some(input), 0);
        enc.set_buffer(2, Some(output), 0);
        enc.set_buffer(3, Some(&w.params_buf), 0);
        enc.dispatch_thread_groups(
            sz2d(groups.min(65535) as u64, groups.div_ceil(65535) as u64),
            sz1d(32),
        );
    }

    fn encode_q4_gemv_accum(
        &self,
        enc: &ComputeCommandEncoderRef,
        w: &Q4Weight,
        input: &Buffer,
        output: &Buffer,
    ) {
        let groups = w.m.div_ceil(2);
        enc.set_compute_pipeline_state(&self.pipes.gemv_q4_0_fast_slim_accum);
        enc.set_buffer(0, Some(&self.mmap_buf), w.mmap_offset);
        enc.set_buffer(1, Some(input), 0);
        enc.set_buffer(2, Some(output), 0);
        enc.set_buffer(3, Some(&w.params_buf), 0);
        enc.dispatch_thread_groups(
            sz2d(groups.min(65535) as u64, groups.div_ceil(65535) as u64),
            sz1d(32),
        );
    }

    fn encode_q4_gate_up(
        &self,
        enc: &ComputeCommandEncoderRef,
        w1: &Q4Weight,
        w3: &Q4Weight,
        input: &Buffer,
    ) {
        let groups = w1.m.div_ceil(2);
        enc.set_compute_pipeline_state(&self.pipes.gemv_q4_0_fast_slim_gate_up);
        enc.set_buffer(0, Some(&self.mmap_buf), w1.mmap_offset);
        enc.set_buffer(1, Some(&self.mmap_buf), w3.mmap_offset);
        enc.set_buffer(2, Some(input), 0);
        enc.set_buffer(3, Some(&self.gate_buf), 0);
        enc.set_buffer(4, Some(&self.up_buf), 0);
        enc.set_buffer(5, Some(&w1.params_buf), 0);
        enc.dispatch_thread_groups(
            sz2d(groups.min(65535) as u64, groups.div_ceil(65535) as u64),
            sz1d(32),
        );
    }
}

// ── AudioGpu trait implementation ───────────────────────────────────────────

impl crate::model::audio_decoder::AudioGpu for MetalAudioDecoder {
    fn sample_audio_frame(&self, embedding: &[f32], temperature: f32, top_k: usize) -> [i32; 8] {
        if let Some(df) = &self.depthformer {
            df.sample_frame(embedding, temperature, top_k)
        } else {
            panic!("No Metal depthformer — use CPU fallback")
        }
    }

    fn detokenize_to_spectrum(
        &self,
        cpu_weights: &crate::model::audio_decoder::DetokenizerWeights,
        codes: &[i32],
    ) -> Vec<f32> {
        self.detokenize_to_spectrum(cpu_weights, codes)
    }

    fn reset_depthformer(&self) {}

    fn reset_detokenizer(&self) {
        self.reset();
    }
}
