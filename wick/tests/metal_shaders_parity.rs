//! Parity tests for native MSL shaders vs CPU reference implementations.
//!
//! Run with:
//!   cargo test -p wick --features metal --test metal_shaders_parity

#![cfg(all(feature = "metal", target_os = "macos"))]

use metal::MTLSize;
use wick::backend::metal::{MetalContext, shaders};

fn tg_size(w: u64) -> MTLSize {
    MTLSize {
        width: w,
        height: 1,
        depth: 1,
    }
}

fn setup() -> Option<MetalContext> {
    MetalContext::new().ok()
}

fn assert_close(name: &str, a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "{name}: length mismatch");
    let mut max_abs = 0.0f32;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    assert!(
        max_abs < tol,
        "{name}: max_abs={max_abs:.3e} > tol={tol:.3e}"
    );
    eprintln!("{name}: max_abs={max_abs:.3e} OK");
}

fn run_1d(
    ctx: &MetalContext,
    src: &str,
    entry: &str,
    buffers: &[&metal::Buffer],
    grid_x: u64,
    tg_x: u64,
    use_threads: bool,
) {
    let pipeline = ctx.create_pipeline(src, entry).expect("compile");
    let cb = ctx.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    for (i, b) in buffers.iter().enumerate() {
        enc.set_buffer(i as u64, Some(b), 0);
    }
    if use_threads {
        enc.dispatch_threads(tg_size(grid_x), tg_size(tg_x));
    } else {
        enc.dispatch_thread_groups(tg_size(grid_x), tg_size(tg_x));
    }
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
}

#[test]
fn test_add_inplace() {
    let Some(ctx) = setup() else { return };
    let n = 1000u32;
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.2).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

    let a_buf = ctx.upload_f32(&a);
    let b_buf = ctx.upload_f32(&b);
    let params = [n, 0u32];
    let p_buf = ctx.upload_bytes(bytemuck::cast_slice(&params));
    run_1d(
        &ctx,
        shaders::ELEMENTWISE,
        "add_inplace",
        &[&a_buf, &b_buf, &p_buf],
        n as u64,
        256,
        true,
    );
    let got = ctx.read_f32(&a_buf, n as usize);
    assert_close("add_inplace", &expected, &got, 1e-5);
}

#[test]
fn test_mul_inplace() {
    let Some(ctx) = setup() else { return };
    let n = 1000u32;
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.2).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * y).collect();

    let a_buf = ctx.upload_f32(&a);
    let b_buf = ctx.upload_f32(&b);
    let p_buf = ctx.upload_bytes(bytemuck::cast_slice(&[n, 0u32]));
    run_1d(
        &ctx,
        shaders::ELEMENTWISE,
        "mul_inplace",
        &[&a_buf, &b_buf, &p_buf],
        n as u64,
        256,
        true,
    );
    let got = ctx.read_f32(&a_buf, n as usize);
    assert_close("mul_inplace", &expected, &got, 1e-5);
}

#[test]
fn test_silu_mul_inplace() {
    let Some(ctx) = setup() else { return };
    let n = 1000u32;
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.01 - 5.0).collect();
    let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let expected: Vec<f32> = a
        .iter()
        .zip(&b)
        .map(|(g, y)| (g / (1.0 + (-g).exp())) * y)
        .collect();

    let a_buf = ctx.upload_f32(&a);
    let b_buf = ctx.upload_f32(&b);
    let p_buf = ctx.upload_bytes(bytemuck::cast_slice(&[n, 0u32]));
    run_1d(
        &ctx,
        shaders::ELEMENTWISE,
        "silu_mul_inplace",
        &[&a_buf, &b_buf, &p_buf],
        n as u64,
        256,
        true,
    );
    let got = ctx.read_f32(&a_buf, n as usize);
    assert_close("silu_mul_inplace", &expected, &got, 1e-4);
}

#[test]
fn test_rmsnorm() {
    let Some(ctx) = setup() else { return };
    let n = 2048u32;
    let eps = 1e-5f32;
    let x: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.031).sin()) * 2.0).collect();
    let w: Vec<f32> = (0..n)
        .map(|i| 0.5 + (i as f32 * 0.017).cos() * 0.5)
        .collect();
    let mut expected = x.clone();
    let sum_sq: f32 = expected.iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
    for i in 0..expected.len() {
        expected[i] = expected[i] * inv_rms * w[i];
    }

    let x_buf = ctx.upload_f32(&x);
    let dst_buf = ctx.create_buffer((n as u64) * 4);
    let w_buf = ctx.upload_f32(&w);
    let params = [n, eps.to_bits(), 0u32, 0u32];
    let p_buf = ctx.upload_bytes(bytemuck::cast_slice(&params));
    run_1d(
        &ctx,
        shaders::RMSNORM,
        "rmsnorm",
        &[&x_buf, &dst_buf, &w_buf, &p_buf],
        1,
        256,
        false,
    );
    let got = ctx.read_f32(&dst_buf, n as usize);
    assert_close("rmsnorm", &expected, &got, 1e-4);
}

#[test]
fn test_per_head_rmsnorm() {
    let Some(ctx) = setup() else { return };
    let head_dim = 128u32;
    let n_heads = 8u32;
    let total = (head_dim * n_heads) as usize;
    let eps = 1e-5f32;
    let x: Vec<f32> = (0..total)
        .map(|i| ((i as f32 * 0.013).sin()) * 2.0)
        .collect();
    let w: Vec<f32> = (0..head_dim)
        .map(|i| 0.8 + (i as f32 * 0.021).cos() * 0.2)
        .collect();

    let mut expected = x.clone();
    for h in 0..n_heads as usize {
        let off = h * head_dim as usize;
        let slice = &expected[off..off + head_dim as usize];
        let sum_sq: f32 = slice.iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (sum_sq / head_dim as f32 + eps).sqrt();
        for i in 0..head_dim as usize {
            expected[off + i] = expected[off + i] * inv_rms * w[i];
        }
    }

    let x_buf = ctx.upload_f32(&x);
    let w_buf = ctx.upload_f32(&w);
    let params = [head_dim, eps.to_bits(), 0u32, 0u32];
    let p_buf = ctx.upload_bytes(bytemuck::cast_slice(&params));
    run_1d(
        &ctx,
        shaders::PER_HEAD_RMSNORM,
        "per_head_rmsnorm",
        &[&x_buf, &w_buf, &p_buf],
        n_heads as u64,
        256,
        false,
    );
    let got = ctx.read_f32(&x_buf, total);
    assert_close("per_head_rmsnorm", &expected, &got, 1e-4);
}

#[test]
fn test_softmax() {
    let Some(ctx) = setup() else { return };
    let n = 4096u32;
    let x: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.0021).sin()) * 3.0).collect();
    let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|v| (v - max_val).exp()).collect();
    let sum_e: f32 = exps.iter().sum();
    let expected: Vec<f32> = exps.iter().map(|e| e / sum_e).collect();

    let x_buf = ctx.upload_f32(&x);
    let p_buf = ctx.upload_bytes(bytemuck::cast_slice(&[n, 0u32]));
    run_1d(
        &ctx,
        shaders::SOFTMAX,
        "softmax",
        &[&x_buf, &p_buf],
        1,
        256,
        false,
    );
    let got = ctx.read_f32(&x_buf, n as usize);
    assert_close("softmax", &expected, &got, 1e-5);
}

#[test]
fn test_gemv_f32() {
    let Some(ctx) = setup() else { return };
    let m = 128u32;
    let k = 256u32;
    let a: Vec<f32> = (0..(m * k) as usize)
        .map(|i| ((i as f32 * 0.007).sin()) * 0.5)
        .collect();
    let x: Vec<f32> = (0..k).map(|i| ((i as f32 * 0.013).cos()) * 0.5).collect();
    let mut expected = vec![0.0f32; m as usize];
    for row in 0..m as usize {
        let mut s = 0.0f32;
        for col in 0..k as usize {
            s += a[row * k as usize + col] * x[col];
        }
        expected[row] = s;
    }

    let a_buf = ctx.upload_f32(&a);
    let x_buf = ctx.upload_f32(&x);
    let y_buf = ctx.create_buffer((m as u64) * 4);
    let p_buf = ctx.upload_bytes(bytemuck::cast_slice(&[m, k]));

    let pipeline = ctx
        .create_pipeline(shaders::GEMV_F32, "gemv_f32")
        .expect("compile");
    let cb = ctx.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&a_buf), 0);
    enc.set_buffer(1, Some(&x_buf), 0);
    enc.set_buffer(2, Some(&y_buf), 0);
    enc.set_buffer(3, Some(&p_buf), 0);
    let tg_count = MTLSize {
        width: (m as u64).min(65535),
        height: (m as u64).div_ceil(65535),
        depth: 1,
    };
    enc.dispatch_thread_groups(tg_count, tg_size(32));
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    let got = ctx.read_f32(&y_buf, m as usize);
    assert_close("gemv_f32", &expected, &got, 1e-3);
}

#[test]
fn test_rope() {
    let Some(ctx) = setup() else { return };
    let n_heads = 4u32;
    let n_kv_heads = 2u32;
    let head_dim = 64u32;
    let pos = 7u32;
    let freq_base = 10000.0f32;

    let q: Vec<f32> = (0..(n_heads * head_dim) as usize)
        .map(|i| (i as f32 * 0.011).sin())
        .collect();
    let k: Vec<f32> = (0..(n_kv_heads * head_dim) as usize)
        .map(|i| (i as f32 * 0.017).cos())
        .collect();

    let apply_rope = |buf: &mut [f32], heads: u32| {
        let half = head_dim as usize / 2;
        for h in 0..heads as usize {
            for d in 0..half {
                let freq = 1.0 / freq_base.powf(2.0 * d as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                let (s, c) = angle.sin_cos();
                let i0 = h * head_dim as usize + d;
                let i1 = i0 + half;
                let x0 = buf[i0];
                let x1 = buf[i1];
                buf[i0] = x0 * c - x1 * s;
                buf[i1] = x0 * s + x1 * c;
            }
        }
    };
    let mut q_exp = q.clone();
    let mut k_exp = k.clone();
    apply_rope(&mut q_exp, n_heads);
    apply_rope(&mut k_exp, n_kv_heads);

    let q_buf = ctx.upload_f32(&q);
    let k_buf = ctx.upload_f32(&k);
    let params = [pos, n_heads, n_kv_heads, head_dim, freq_base.to_bits()];
    let p_buf = ctx.upload_bytes(bytemuck::cast_slice(&params));

    let max_pairs = n_heads.max(n_kv_heads) * head_dim / 2;
    run_1d(
        &ctx,
        shaders::ROPE,
        "rope",
        &[&q_buf, &k_buf, &p_buf],
        max_pairs as u64,
        256,
        true,
    );
    let q_got = ctx.read_f32(&q_buf, q.len());
    let k_got = ctx.read_f32(&k_buf, k.len());
    assert_close("rope-Q", &q_exp, &q_got, 1e-5);
    assert_close("rope-K", &k_exp, &k_got, 1e-5);
}

#[test]
fn test_attention() {
    let Some(ctx) = setup() else { return };
    let n_heads = 4u32;
    let n_kv_heads = 2u32;
    let head_dim = 32u32;
    let kv_dim = n_kv_heads * head_dim;
    let seq_len = 16u32;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q: Vec<f32> = (0..(n_heads * head_dim) as usize)
        .map(|i| ((i as f32 * 0.013).sin()) * 0.3)
        .collect();
    let k: Vec<f32> = (0..(seq_len * kv_dim) as usize)
        .map(|i| ((i as f32 * 0.007).cos()) * 0.3)
        .collect();
    let v: Vec<f32> = (0..(seq_len * kv_dim) as usize)
        .map(|i| ((i as f32 * 0.019).sin()) * 0.3)
        .collect();

    // CPU reference.
    let group_size = n_heads / n_kv_heads;
    let mut expected = vec![0.0f32; (n_heads * head_dim) as usize];
    for h in 0..n_heads as usize {
        let kv_h = h / group_size as usize;
        let kv_off = kv_h * head_dim as usize;
        let q_off = h * head_dim as usize;
        let mut scores = vec![0.0f32; seq_len as usize];
        for t in 0..seq_len as usize {
            let mut dot = 0.0f32;
            for d in 0..head_dim as usize {
                dot += q[q_off + d] * k[t * kv_dim as usize + kv_off + d];
            }
            scores[t] = dot * scale;
        }
        let mx = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - mx).exp();
            sum += *s;
        }
        for s in scores.iter_mut() {
            *s /= sum;
        }
        for d in 0..head_dim as usize {
            let mut val = 0.0f32;
            for t in 0..seq_len as usize {
                val += scores[t] * v[t * kv_dim as usize + kv_off + d];
            }
            expected[q_off + d] = val;
        }
    }

    let q_buf = ctx.upload_f32(&q);
    let k_buf = ctx.upload_f32(&k);
    let v_buf = ctx.upload_f32(&v);
    let out_buf = ctx.create_buffer((n_heads * head_dim) as u64 * 4);
    let params = [
        n_heads,
        n_kv_heads,
        head_dim,
        kv_dim,
        seq_len,
        scale.to_bits(),
        0u32,
        0u32,
    ];
    let p_buf = ctx.upload_bytes(bytemuck::cast_slice(&params));
    run_1d(
        &ctx,
        shaders::ATTENTION,
        "attention",
        &[&q_buf, &k_buf, &v_buf, &out_buf, &p_buf],
        n_heads as u64,
        256,
        false,
    );
    let got = ctx.read_f32(&out_buf, (n_heads * head_dim) as usize);
    assert_close("attention", &expected, &got, 1e-4);
}

/// Q6_K GEMV parity: synthesize Q6_K blocks, dequantize on CPU, GEMV on GPU,
/// compare. Covers the vocab-head use case (m large, k a multiple of 256).
#[test]
fn test_gemv_q6_k() {
    use wick::quant::{BlockQ6K, dequantize_q6_k_block};
    let Some(ctx) = setup() else { return };
    let m = 512u32;
    let k = 512u32; // = 2 super-blocks per row
    let qk_k = 256usize;
    let nb = k as usize / qk_k;

    // Build synthetic Q6_K weights: deterministic nibble + qh + scales.
    let mut raw = Vec::with_capacity(m as usize * nb * 210);
    let mut expected_f32 = vec![0.0f32; m as usize * k as usize];
    for row in 0..m as usize {
        for b in 0..nb {
            let mut blk = BlockQ6K {
                ql: [0u8; 128],
                qh: [0u8; 64],
                scales: [0i8; 16],
                d: half::f16::from_f32(0.01 + (row as f32 * 0.003).sin() * 0.002).to_bits(),
            };
            for i in 0..128 {
                blk.ql[i] = ((row * 37 + b * 13 + i) & 0xFF) as u8;
            }
            for i in 0..64 {
                blk.qh[i] = ((row * 11 + b * 7 + i) & 0xFF) as u8;
            }
            for i in 0..16 {
                blk.scales[i] = (((row * 3 + b * 5 + i) as i32 & 0x7F) - 32) as i8;
            }
            let dq = dequantize_q6_k_block(&blk);
            let row_off = row * k as usize + b * qk_k;
            expected_f32[row_off..row_off + qk_k].copy_from_slice(&dq);
            // Serialize block into raw byte buffer (matches shader's layout).
            raw.extend_from_slice(&blk.ql);
            raw.extend_from_slice(&blk.qh);
            let sc_bytes: &[u8] = bytemuck::cast_slice(&blk.scales);
            raw.extend_from_slice(sc_bytes);
            raw.extend_from_slice(&blk.d.to_le_bytes());
        }
    }
    raw.extend_from_slice(&[0u8; 16]); // safety pad

    // x vector.
    let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.013).sin()).collect();

    // CPU reference: y[r] = Σ weight_f32[r][i] × x[i].
    let mut y_ref = vec![0.0f32; m as usize];
    for row in 0..m as usize {
        let mut s = 0.0f32;
        for i in 0..k as usize {
            s += expected_f32[row * k as usize + i] * x[i];
        }
        y_ref[row] = s;
    }

    // GPU dispatch.
    let a_buf = ctx.upload_bytes(&raw);
    let x_buf = ctx.upload_f32(&x);
    let y_buf = ctx.create_buffer((m as u64) * 4);
    let p_buf = ctx.upload_bytes(bytemuck::cast_slice(&[m, k]));
    let pl = ctx
        .create_pipeline(shaders::GEMV_Q6_K, "gemv_q6_k")
        .unwrap();
    let cb = ctx.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pl);
    enc.set_buffer(0, Some(&a_buf), 0);
    enc.set_buffer(1, Some(&x_buf), 0);
    enc.set_buffer(2, Some(&y_buf), 0);
    enc.set_buffer(3, Some(&p_buf), 0);
    enc.dispatch_thread_groups(
        metal::MTLSize {
            width: (m / 4) as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        },
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    let y_gpu = ctx.read_f32(&y_buf, m as usize);
    assert_close("gemv_q6_k", &y_ref, &y_gpu, 5e-3);
}

/// Fast Q4_0 GEMV (llama.cpp-style) parity vs classic gemv_q4_0.
#[test]
fn test_gemv_q4_0_fast() {
    let Some(ctx) = setup() else { return };
    // Covers attention projection shapes and FFN shapes used by LFM2.
    for &(m, k) in &[(256u32, 1024u32), (1024, 1024), (2048, 1024), (1024, 2048)] {
        let nb = (k / 32) as usize;
        // Deterministic pseudo-random weight bytes.
        let mut q4: Vec<u8> = Vec::with_capacity(m as usize * nb * 18);
        for i in 0..(m as usize * nb) {
            // 2-byte f16 scale + 16 bytes of nibbles
            let d = 0.05f32 + (i as f32 * 0.0017).sin() * 0.02;
            let d_f16 = half::f16::from_f32(d);
            q4.extend_from_slice(&d_f16.to_bits().to_le_bytes());
            for j in 0..16 {
                q4.push(((i * 17 + j * 53) & 0xFF) as u8);
            }
        }
        q4.extend_from_slice(&[0u8; 8]);
        let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013).sin()).collect();

        let a_buf = ctx.upload_bytes(&q4);
        let x_buf = ctx.upload_f32(&x);
        let y_ref = ctx.create_buffer((m as u64) * 4);
        let y_fast = ctx.create_buffer((m as u64) * 4);
        let params = [m, k];
        let p_buf = ctx.upload_bytes(bytemuck::cast_slice(&params));

        // Classic gemv_q4_0: ROWS_PER_TG=2, 32 threads.
        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        let pl = ctx
            .create_pipeline(shaders::GEMV_Q4_0, "gemv_q4_0")
            .unwrap();
        enc.set_compute_pipeline_state(&pl);
        enc.set_buffer(0, Some(&a_buf), 0);
        enc.set_buffer(1, Some(&x_buf), 0);
        enc.set_buffer(2, Some(&y_ref), 0);
        enc.set_buffer(3, Some(&p_buf), 0);
        enc.dispatch_thread_groups(
            metal::MTLSize {
                width: m.div_ceil(2) as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        // Fast gemv_q4_0_fast: ROWS_PER_TG=8, 64 threads.
        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        let pl = ctx
            .create_pipeline(shaders::GEMV_Q4_0_FAST, "gemv_q4_0_fast")
            .unwrap();
        enc.set_compute_pipeline_state(&pl);
        enc.set_buffer(0, Some(&a_buf), 0);
        enc.set_buffer(1, Some(&x_buf), 0);
        enc.set_buffer(2, Some(&y_fast), 0);
        enc.set_buffer(3, Some(&p_buf), 0);
        enc.dispatch_thread_groups(
            metal::MTLSize {
                width: m.div_ceil(8) as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let ref_vals = ctx.read_f32(&y_ref, m as usize);
        let fast_vals = ctx.read_f32(&y_fast, m as usize);
        let mut max_abs = 0f32;
        let mut max_rel = 0f32;
        for i in 0..m as usize {
            let d = (ref_vals[i] - fast_vals[i]).abs();
            max_abs = max_abs.max(d);
            let denom = ref_vals[i].abs().max(1e-6);
            max_rel = max_rel.max(d / denom);
        }
        assert!(
            max_abs < 1e-2 || max_rel < 1e-3,
            "m={m} k={k}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
        );
        eprintln!("gemv_q4_0_fast m={m} k={k}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}");
    }
}

/// FlashAttention: same bindings as attention, tiled with online softmax.
/// Covers tile boundary crossings (seq_len = 16, 64, 100, 513).
#[test]
fn test_flash_attention() {
    let Some(ctx) = setup() else { return };
    for &seq_len_u in &[16u32, 64, 100, 513] {
        let n_heads = 4u32;
        let n_kv_heads = 2u32;
        let head_dim = 64u32; // must match realistic LFM2 head_dim for par coverage
        let kv_dim = n_kv_heads * head_dim;
        let seq_len = seq_len_u;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q: Vec<f32> = (0..(n_heads * head_dim) as usize)
            .map(|i| ((i as f32 * 0.013).sin()) * 0.3)
            .collect();
        let k: Vec<f32> = (0..(seq_len * kv_dim) as usize)
            .map(|i| ((i as f32 * 0.007).cos()) * 0.3)
            .collect();
        let v: Vec<f32> = (0..(seq_len * kv_dim) as usize)
            .map(|i| ((i as f32 * 0.019).sin()) * 0.3)
            .collect();

        // CPU reference.
        let group_size = n_heads / n_kv_heads;
        let mut expected = vec![0.0f32; (n_heads * head_dim) as usize];
        for h in 0..n_heads as usize {
            let kv_h = h / group_size as usize;
            let kv_off = kv_h * head_dim as usize;
            let q_off = h * head_dim as usize;
            let mut scores = vec![0.0f32; seq_len as usize];
            for t in 0..seq_len as usize {
                let mut dot = 0.0f32;
                for d in 0..head_dim as usize {
                    dot += q[q_off + d] * k[t * kv_dim as usize + kv_off + d];
                }
                scores[t] = dot * scale;
            }
            let mx = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - mx).exp();
                sum += *s;
            }
            for s in scores.iter_mut() {
                *s /= sum;
            }
            for d in 0..head_dim as usize {
                let mut val = 0.0f32;
                for t in 0..seq_len as usize {
                    val += scores[t] * v[t * kv_dim as usize + kv_off + d];
                }
                expected[q_off + d] = val;
            }
        }

        let q_buf = ctx.upload_f32(&q);
        let k_buf = ctx.upload_f32(&k);
        let v_buf = ctx.upload_f32(&v);
        let out_buf = ctx.create_buffer((n_heads * head_dim) as u64 * 4);
        let params = [
            n_heads,
            n_kv_heads,
            head_dim,
            kv_dim,
            seq_len,
            scale.to_bits(),
            0u32,
            0u32,
        ];
        let p_buf = ctx.upload_bytes(bytemuck::cast_slice(&params));
        run_1d(
            &ctx,
            shaders::FLASH_ATTENTION,
            "flash_attention",
            &[&q_buf, &k_buf, &v_buf, &out_buf, &p_buf],
            n_heads as u64,
            256,
            false,
        );
        let got = ctx.read_f32(&out_buf, (n_heads * head_dim) as usize);
        assert_close(
            &format!("flash_attention[seq_len={seq_len}]"),
            &expected,
            &got,
            1e-4,
        );
    }
}

#[test]
fn test_conv1d() {
    let Some(ctx) = setup() else { return };
    let hs = 128u32;
    let kernel_size = 3u32;
    let d_conv = kernel_size - 1; // rolling buffer depth

    let input: Vec<f32> = (0..hs).map(|i| (i as f32 * 0.03).sin()).collect();
    let buffer: Vec<f32> = (0..(d_conv * hs))
        .map(|i| (i as f32 * 0.05).cos())
        .collect();
    let weight: Vec<f32> = (0..(hs * kernel_size))
        .map(|i| (i as f32 * 0.011).sin())
        .collect();

    // CPU reference.
    let mut expected = vec![0.0f32; hs as usize];
    for ch in 0..hs as usize {
        let mut sum = 0.0f32;
        for k in 0..d_conv as usize {
            sum += buffer[k * hs as usize + ch] * weight[ch * kernel_size as usize + k];
        }
        sum += input[ch] * weight[ch * kernel_size as usize + d_conv as usize];
        expected[ch] = sum;
    }

    let input_buf = ctx.upload_f32(&input);
    let buffer_buf = ctx.upload_f32(&buffer);
    let weight_buf = ctx.upload_f32(&weight);
    let output_buf = ctx.create_buffer((hs as u64) * 4);
    let params = [hs, kernel_size, d_conv, 0u32];
    let p_buf = ctx.upload_bytes(bytemuck::cast_slice(&params));
    run_1d(
        &ctx,
        shaders::CONV1D,
        "conv1d_depthwise",
        &[&input_buf, &buffer_buf, &weight_buf, &output_buf, &p_buf],
        hs as u64,
        256,
        true,
    );
    let got = ctx.read_f32(&output_buf, hs as usize);
    assert_close("conv1d", &expected, &got, 1e-5);
}
