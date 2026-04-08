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

// ── GPU weight reference ────────────────────────────────────────────────────

struct MetalWeight {
    mmap_offset: u64,
    dtype: DType,
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
    gemv_q4_0_fast_slim: ComputePipelineState,
    gemv_q4_0_fast_slim_accum: ComputePipelineState,
    gemv_q4_0_fast_slim_gate_up: ComputePipelineState,
    memcpy_f32: ComputePipelineState,
    mul_out: ComputePipelineState,
    add_inplace: ComputePipelineState,
    silu_mul_inplace: ComputePipelineState,
    rmsnorm: ComputePipelineState,
    qk_norm_rope: ComputePipelineState,
    attention: ComputePipelineState,
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

    layers: Vec<DetokLayerGpu>,
    output_norm: Buffer,
    lin_w: MetalWeight,
    lin_b: Buffer,

    // Scratch
    hidden_buf: Buffer,
    normed_buf: Buffer,
    proj_buf: Buffer,     // [3 * n_embd] for conv in_proj output
    bx_buf: Buffer,       // [n_embd] for b*x
    conv_out_buf: Buffer, // [n_embd]
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

    mmap_buf: Buffer,
    _mmap: memmap2::Mmap,
}

impl MetalAudioDecoder {
    pub fn from_gguf(gguf: &GgufFile, vocoder_path: &Path) -> Result<Self> {
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

        // Pipelines
        let pipes = Pipelines {
            gemv_q4_0_fast_slim: ctx
                .create_pipeline(shaders::GEMV_Q4_0_FAST, "gemv_q4_0_fast_slim")?,
            gemv_q4_0_fast_slim_accum: ctx
                .create_pipeline(shaders::GEMV_Q4_0_FAST, "gemv_q4_0_fast_slim_accum")?,
            gemv_q4_0_fast_slim_gate_up: ctx
                .create_pipeline(shaders::GEMV_Q4_0_FAST, "gemv_q4_0_fast_slim_gate_up")?,
            memcpy_f32: ctx.create_pipeline(shaders::ELEMENTWISE, "memcpy_f32")?,
            mul_out: ctx.create_pipeline(shaders::ELEMENTWISE, "mul_out")?,
            add_inplace: ctx.create_pipeline(shaders::ELEMENTWISE, "add_inplace")?,
            silu_mul_inplace: ctx.create_pipeline(shaders::ELEMENTWISE, "silu_mul_inplace")?,
            rmsnorm: ctx.create_pipeline(shaders::RMSNORM, "rmsnorm")?,
            qk_norm_rope: ctx.create_pipeline(shaders::QK_NORM_ROPE, "qk_norm_rope")?,
            attention: ctx.create_pipeline(shaders::ATTENTION, "attention")?,
            conv1d: ctx.create_pipeline(shaders::CONV1D, "conv1d_depthwise")?,
        };

        // Mmap for zero-copy weight access
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

        // Weight upload helper
        let make_weight = |name: &str| -> Result<MetalWeight> {
            let (off, rows, cols, dtype) = gguf.tensor_meta(name)?;
            let params_buf = ctx.upload_bytes(bytemuck::cast_slice(&[rows as u32, cols as u32]));
            Ok(MetalWeight {
                mmap_offset: off as u64,
                dtype,
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
                kv_k[i] = Some(ab(cfg.swa_window_size * kv_dim));
                kv_v[i] = Some(ab(cfg.swa_window_size * kv_dim));
            }
        }

        let hidden_buf = ab(n_embd);
        let normed_buf = ab(n_embd);
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

        Ok(Self {
            ctx,
            cfg,
            pipes,
            params,
            layers,
            output_norm,
            lin_w,
            lin_b,
            hidden_buf,
            normed_buf,
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
            mmap_buf,
            _mmap: mmap,
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

    /// Minimal GPU smoke test: write known data to a buffer, run one rmsnorm,
    /// read back. Returns the first 4 output values.
    pub fn smoke_test(&self) -> Vec<f32> {
        let hs = self.cfg.n_embd;

        // Write ones into hidden_buf
        unsafe {
            let dst = std::slice::from_raw_parts_mut(self.hidden_buf.contents() as *mut f32, hs);
            for v in dst.iter_mut() {
                *v = 1.0;
            }
        }

        // Run rmsnorm → normed_buf
        let cb = self.ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        self.encode_rmsnorm_out(enc, &self.hidden_buf, &self.normed_buf, &self.output_norm);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        // Read back
        let out =
            unsafe { std::slice::from_raw_parts(self.normed_buf.contents() as *const f32, hs) };
        eprintln!(
            "[smoke] rmsnorm output first4: [{:.6}, {:.6}, {:.6}, {:.6}]",
            out[0], out[1], out[2], out[3]
        );
        out[..4].to_vec()
    }

    /// Test full 8-layer dispatch for 6 tokens, one layer per command buffer.
    pub fn smoke_test_full(&self) {
        let hs = self.cfg.n_embd;
        let n_frames = 6;

        // Write test data
        unsafe {
            let dst = std::slice::from_raw_parts_mut(
                self.tokens_buf.contents() as *mut f32,
                n_frames * hs,
            );
            for (i, v) in dst.iter_mut().enumerate() {
                *v = (i as f32 * 0.001).sin() * 0.1;
            }
        }
        for buf in &self.conv_bufs {
            if let Some(b) = buf {
                unsafe {
                    std::ptr::write_bytes(b.contents() as *mut u8, 0, b.length() as usize);
                }
            }
        }
        self.n_past.set(0);
        let n_past = 0;

        for (il, lw) in self.layers.iter().enumerate() {
            let t0 = std::time::Instant::now();
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
            let dt = t0.elapsed();
            let kind = if self.cfg.layer_is_conv[il] {
                "conv"
            } else {
                "attn"
            };
            eprintln!(
                "  [smoke_full] layer {il} ({kind}): {:.1}ms",
                dt.as_secs_f64() * 1000.0
            );
        }
    }

    /// Test one full conv layer dispatch for 1 token.
    pub fn smoke_test_conv_layer(&self) -> Vec<f32> {
        let hs = self.cfg.n_embd;

        // Write test data into tokens_buf[0..hs]
        unsafe {
            let dst = std::slice::from_raw_parts_mut(self.tokens_buf.contents() as *mut f32, hs);
            for (i, v) in dst.iter_mut().enumerate() {
                *v = (i as f32 * 0.001).sin();
            }
        }
        // Zero conv buffer
        if let Some(b) = &self.conv_bufs[0] {
            unsafe {
                std::ptr::write_bytes(b.contents() as *mut u8, 0, b.length() as usize);
            }
        }

        let cb = self.ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        eprintln!("  [smoke_conv] dispatching layer 0, 1 token...");
        self.encode_conv_layer(enc, &self.layers[0], 0, 1);
        enc.end_encoding();
        eprintln!("  [smoke_conv] committing...");
        cb.commit();
        cb.wait_until_completed();
        eprintln!("  [smoke_conv] done");

        let out =
            unsafe { std::slice::from_raw_parts(self.tokens_buf.contents() as *const f32, hs) };
        eprintln!(
            "[smoke] conv layer 0 output first4: [{:.6}, {:.6}, {:.6}, {:.6}]",
            out[0], out[1], out[2], out[3]
        );
        out[..4].to_vec()
    }

    /// Test one GEMV dispatch and return first 4 values.
    pub fn smoke_test_gemv(&self) -> Vec<f32> {
        let hs = self.cfg.n_embd;

        // Write ones into normed_buf
        unsafe {
            let dst = std::slice::from_raw_parts_mut(self.normed_buf.contents() as *mut f32, hs);
            for v in dst.iter_mut() {
                *v = 0.01;
            }
        }

        // Run conv_in_proj GEMV (layer 0, which is conv)
        let lw = &self.layers[0];
        let w = lw.conv_in_proj.as_ref().unwrap();

        let cb = self.ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        self.encode_gemv(enc, w, &self.normed_buf, &self.proj_buf, false);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let out =
            unsafe { std::slice::from_raw_parts(self.proj_buf.contents() as *const f32, 3 * hs) };
        eprintln!(
            "[smoke] gemv output first4: [{:.6}, {:.6}, {:.6}, {:.6}]",
            out[0], out[1], out[2], out[3]
        );
        out[..4].to_vec()
    }

    // ── GEMV dispatch ───────────────────────────────────────────────────

    fn encode_gemv(
        &self,
        enc: &ComputeCommandEncoderRef,
        w: &MetalWeight,
        input: &Buffer,
        output: &Buffer,
        accum: bool,
    ) {
        let pipe = if accum {
            &self.pipes.gemv_q4_0_fast_slim_accum
        } else {
            &self.pipes.gemv_q4_0_fast_slim
        };
        let groups = w.m.div_ceil(2);
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&self.mmap_buf), w.mmap_offset);
        enc.set_buffer(1, Some(input), 0);
        enc.set_buffer(2, Some(output), 0);
        enc.set_buffer(3, Some(&w.params_buf), 0);
        enc.dispatch_thread_groups(
            sz2d(groups.min(65535) as u64, groups.div_ceil(65535) as u64),
            sz1d(32),
        );
    }

    fn encode_gemv_gate_up(
        &self,
        enc: &ComputeCommandEncoderRef,
        w1: &MetalWeight,
        w3: &MetalWeight,
        input: &Buffer,
        gate: &Buffer,
        up: &Buffer,
    ) {
        let groups = w1.m.div_ceil(2);
        enc.set_compute_pipeline_state(&self.pipes.gemv_q4_0_fast_slim_gate_up);
        enc.set_buffer(0, Some(&self.mmap_buf), w1.mmap_offset);
        enc.set_buffer(1, Some(input), 0);
        enc.set_buffer(2, Some(gate), 0);
        enc.set_buffer(3, Some(&w1.params_buf), 0);
        enc.set_buffer(4, Some(&self.mmap_buf), w3.mmap_offset);
        enc.set_buffer(5, Some(up), 0);
        enc.dispatch_thread_groups(
            sz2d(groups.min(65535) as u64, groups.div_ceil(65535) as u64),
            sz1d(32),
        );
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
    // If correctness issues arise, add per-buffer barriers via memory_barrier_with_resources.
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
            self.encode_gemv(enc, cin, &self.normed_buf, &self.proj_buf, false);
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
            self.encode_gemv(enc, cop, &self.bx_buf, &self.hidden_buf, true);
            self.barrier(enc);

            // FFN: rmsnorm → gate+up → silu_mul → down → residual
            self.encode_rmsnorm_out(enc, &self.hidden_buf, &self.normed_buf, &lw.ffn_norm);
            self.barrier(enc);
            self.encode_gemv_gate_up(
                enc,
                &lw.ffn_w1,
                &lw.ffn_w3,
                &self.normed_buf,
                &self.gate_buf,
                &self.up_buf,
            );
            self.barrier(enc);
            self.encode_silu_mul(enc);
            self.barrier(enc);
            // down: accumulate into hidden_buf (FFN residual add)
            self.encode_gemv(enc, &lw.ffn_w2, &self.gate_buf, &self.hidden_buf, true);
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

        for t in 0..n_tokens {
            let tok_off = (t * hs * 4) as u64;
            let pos = n_past + t;

            // Load token
            self.encode_memcpy(enc, &self.tokens_buf, tok_off, &self.hidden_buf, 0, hs);
            self.barrier(enc);

            // RMSnorm
            self.encode_rmsnorm_out(enc, &self.hidden_buf, &self.normed_buf, &lw.operator_norm);
            self.barrier(enc);

            // Q/K/V projections
            self.encode_gemv(enc, wq, &self.normed_buf, &self.q_buf, false);
            self.encode_gemv(enc, wk, &self.normed_buf, &self.k_buf, false);
            self.encode_gemv(enc, wv, &self.normed_buf, &self.v_buf, false);
            self.barrier(enc);

            // QK norm + RoPE (NeoX style for detokenizer)
            let freq_bits = self.cfg.rope_freq_base.to_bits();
            let eps_bits = self.cfg.rms_norm_eps.to_bits();
            let rope_params: [u32; 6] = [pos as u32, n_heads, n_kv, hd as u32, eps_bits, freq_bits];
            enc.set_compute_pipeline_state(&self.pipes.qk_norm_rope);
            enc.set_buffer(0, Some(&self.q_buf), 0);
            enc.set_buffer(1, Some(&self.k_buf), 0);
            enc.set_buffer(2, Some(qn), 0);
            enc.set_buffer(3, Some(kn), 0);
            enc.set_bytes(
                4,
                std::mem::size_of_val(&rope_params) as u64,
                rope_params.as_ptr() as *const _,
            );
            enc.dispatch_thread_groups(sz1d((n_heads + n_kv) as u64), sz1d(256));
            self.barrier(enc);

            // Write K, V to cache (ring buffer: pos % swa_window_size)
            let cache_pos = pos % self.cfg.swa_window_size;
            let cache_off = (cache_pos * kv_dim * 4) as u64;
            self.encode_memcpy(enc, &self.k_buf, 0, k_cache, cache_off, kv_dim);
            self.encode_memcpy(enc, &self.v_buf, 0, v_cache, cache_off, kv_dim);
            self.barrier(enc);

            // Attention
            let seq_len = (pos + 1).min(self.cfg.swa_window_size);
            let scale = 1.0f32 / (hd as f32).sqrt();
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
            enc.set_compute_pipeline_state(&self.pipes.attention);
            enc.set_buffer(0, Some(&self.q_buf), 0);
            enc.set_buffer(1, Some(k_cache), 0);
            enc.set_buffer(2, Some(v_cache), 0);
            enc.set_buffer(3, Some(&self.attn_out_buf), 0);
            enc.set_bytes(
                4,
                std::mem::size_of_val(&attn_params) as u64,
                attn_params.as_ptr() as *const _,
            );
            enc.dispatch_thread_groups(sz1d(n_heads as u64), sz1d(256));
            self.barrier(enc);

            // Output projection (accumulate into hidden_buf as residual)
            self.encode_gemv(enc, wo, &self.attn_out_buf, &self.hidden_buf, true);
            self.barrier(enc);

            // FFN
            self.encode_rmsnorm_out(enc, &self.hidden_buf, &self.normed_buf, &lw.ffn_norm);
            self.barrier(enc);
            self.encode_gemv_gate_up(
                enc,
                &lw.ffn_w1,
                &lw.ffn_w3,
                &self.normed_buf,
                &self.gate_buf,
                &self.up_buf,
            );
            self.barrier(enc);
            self.encode_silu_mul(enc);
            self.barrier(enc);
            // down projection accumulates into hidden_buf (residual add)
            self.encode_gemv(enc, &lw.ffn_w2, &self.gate_buf, &self.hidden_buf, true);
            self.barrier(enc);

            // Write back
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

        eprintln!("[gpu_detok] entering frame, n_past={}", self.n_past.get());
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

        eprintln!("[gpu_detok] tokens uploaded, starting layers");
        // 3. Encode layers
        let n_past = self.n_past.get();
        static DETOK_CALL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let call = DETOK_CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        for (il, lw) in self.layers.iter().enumerate() {
            let t0 = std::time::Instant::now();
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
            if call == 0 {
                eprintln!(
                    "[gpu_detok] layer {il}: {:.1}ms",
                    t0.elapsed().as_secs_f64() * 1000.0
                );
            }
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
                // lin_w GEMV
                enc.set_compute_pipeline_state(&self.pipes.gemv_q4_0_fast_slim);
                enc.set_buffer(0, Some(&self.mmap_buf), self.lin_w.mmap_offset);
                enc.set_buffer(1, Some(&self.hidden_buf), 0);
                enc.set_buffer(2, Some(&self.spectrum_buf), spec_off);
                enc.set_buffer(3, Some(&self.lin_w.params_buf), 0);
                let groups = self.lin_w.m.div_ceil(2);
                enc.dispatch_thread_groups(
                    sz2d(groups.min(65535) as u64, groups.div_ceil(65535) as u64),
                    sz1d(32),
                );
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
        if call == 0 {
            let p = unsafe {
                std::slice::from_raw_parts(self.spectrum_buf.contents() as *const f32, 8)
            };
            eprintln!("[gpu_detok] first call spectrum[0..8]: {:?}", p);
            let any_nan = p.iter().any(|v| v.is_nan());
            let any_inf = p.iter().any(|v| v.is_infinite());
            eprintln!("[gpu_detok] nan={any_nan} inf={any_inf}");
        }
        let total = n_frames * spectrum_per_frame;
        let mut spectrum = vec![0.0f32; total];
        unsafe {
            let src = self.spectrum_buf.contents() as *const f32;
            std::ptr::copy_nonoverlapping(src, spectrum.as_mut_ptr(), total);
        }
        spectrum
    }
}
