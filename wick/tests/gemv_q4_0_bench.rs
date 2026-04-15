//! Microbenchmark: WGSL Q4_0 GEMV (via wgpu) vs native MSL Q4_0 GEMV (via metal crate).
//!
//! Run with:
//!   cargo test -p wick --features "gpu metal" --release \
//!     --test gemv_q4_0_bench -- --ignored --nocapture
//!
//! Decides whether porting the wick forward pass to native Metal is justified.

#![cfg(all(feature = "gpu", feature = "metal", target_os = "macos"))]

use wick::backend::{metal::MetalContext, wgpu::GpuContext};

/// Quantize a row-major f32 matrix into Q4_0 blocks (18 bytes per 32 elements).
fn quantize_q4_0(weights_f32: &[f32], m: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % 32, 0);
    let nb = k / 32;
    let mut out = Vec::with_capacity(m * nb * 18);
    for row in 0..m {
        for b in 0..nb {
            let start = row * k + b * 32;
            let block = &weights_f32[start..start + 32];
            let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let d = amax / 7.0;
            let d_f16 = half::f16::from_f32(d);
            out.extend_from_slice(&d_f16.to_bits().to_le_bytes());
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            for qi in 0..16 {
                let lo = ((block[qi] * id + 8.5) as u8).min(15);
                let hi = ((block[16 + qi] * id + 8.5) as u8).min(15);
                out.push(lo | (hi << 4));
            }
        }
    }
    out
}

/// CPU reference GEMV over Q4_0 bytes.
fn q4_0_gemv_cpu(q4: &[u8], x: &[f32], m: usize, k: usize) -> Vec<f32> {
    let nb = k / 32;
    let mut y = vec![0.0f32; m];
    for (row, y_row) in y.iter_mut().enumerate() {
        for b in 0..nb {
            let off = (row * nb + b) * 18;
            let d_bits = u16::from_le_bytes([q4[off], q4[off + 1]]);
            let delta = half::f16::from_bits(d_bits).to_f32();
            for qi in 0..16 {
                let byte = q4[off + 2 + qi];
                let lo = (byte & 0xF) as f32 - 8.0;
                let hi = ((byte >> 4) & 0xF) as f32 - 8.0;
                *y_row += lo * delta * x[b * 32 + qi];
                *y_row += hi * delta * x[b * 32 + qi + 16];
            }
        }
    }
    y
}

/// One shape: (m, k).
struct Shape {
    m: u32,
    k: u32,
    label: &'static str,
}

/// Run one shape on both backends and print results.
fn bench_shape(shape: &Shape, iters: u32) {
    let m = shape.m as usize;
    let k = shape.k as usize;
    let bytes_read = m * (k / 32) * 18;
    println!(
        "\n=== Shape {}: m={}, k={} — weights={:.2} MB, iters={}",
        shape.label,
        shape.m,
        shape.k,
        bytes_read as f64 / 1e6,
        iters
    );

    // Deterministic pseudo-random weights.
    let weights_f32: Vec<f32> = (0..m * k)
        .map(|i| {
            let v = ((i as u32).wrapping_mul(2654435761) >> 16) as f32;
            (v / 32768.0 - 1.0) * 0.5
        })
        .collect();
    let q4_bytes = quantize_q4_0(&weights_f32, m, k);
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013).sin()).collect();
    let expected = q4_0_gemv_cpu(&q4_bytes, &x, m, k);

    // Pad buffer by 8 bytes so both shaders can safely read past the last block.
    let mut q4_padded = q4_bytes.clone();
    q4_padded.extend_from_slice(&[0u8; 8]);

    // ── wgpu (WGSL) path ────────────────────────────────────────────────
    let wgpu_us = match GpuContext::new() {
        Ok(ctx) => Some(bench_wgpu(&ctx, &q4_padded, &x, shape, iters, &expected)),
        Err(e) => {
            eprintln!("wgpu unavailable: {e}");
            None
        }
    };

    // ── native Metal path ───────────────────────────────────────────────
    let metal_us = match MetalContext::new() {
        Ok(ctx) => Some(bench_metal(&ctx, &q4_padded, &x, shape, iters, &expected)),
        Err(e) => {
            eprintln!("metal unavailable: {e}");
            None
        }
    };

    if let (Some(w), Some(mm)) = (wgpu_us, metal_us) {
        let bw_w = bytes_read as f64 / w / 1e9;
        let bw_m = bytes_read as f64 / mm / 1e9;
        println!("wgpu : {:7.1} µs/dispatch  ({:5.1} GB/s)", w * 1e6, bw_w);
        println!(
            "metal: {:7.1} µs/dispatch  ({:5.1} GB/s)   speedup {:.2}×",
            mm * 1e6,
            bw_m,
            w / mm
        );
    }
}

fn bench_wgpu(
    ctx: &GpuContext,
    q4_padded: &[u8],
    x: &[f32],
    shape: &Shape,
    iters: u32,
    expected: &[f32],
) -> f64 {
    let a_buf = ctx.upload_storage(q4_padded, "A_q4");
    let x_buf = ctx.upload_f32(x, "x");
    let y_buf = ctx.create_storage_rw((shape.m as u64) * 4, "y");
    let params = [shape.m, shape.k];
    let params_buf = ctx.upload_storage(bytemuck::cast_slice(&params), "params");

    let pipeline = ctx.create_pipeline(
        wick::backend::wgpu::shaders::GEMV_Q4_0,
        "gemv_q4_0",
        "gemv_q4_0",
    );
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: x_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: y_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    // WGSL shader uses 8 rows/workgroup (not parameterized).
    let groups = shape.m.div_ceil(8);

    // Warmup + correctness check.
    for _ in 0..3 {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
    }
    let result = ctx.download_f32(&y_buf, shape.m as usize);
    check_parity("wgpu", expected, &result);

    // Timed batch — submit all iters, then a single readback to drain.
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
    }
    // Drain the queue.
    let _ = ctx.download_f32(&y_buf, shape.m as usize);
    let elapsed = start.elapsed().as_secs_f64();
    elapsed / iters as f64
}

fn bench_metal(
    ctx: &MetalContext,
    q4_padded: &[u8],
    x: &[f32],
    shape: &Shape,
    iters: u32,
    expected: &[f32],
) -> f64 {
    use metal::MTLSize;

    let a_buf = ctx.upload_bytes(q4_padded);
    let x_buf = ctx.upload_f32(x);
    let y_buf = ctx.create_buffer((shape.m as u64) * 4);
    // Params struct layout: { uint m; uint k; }
    let params = [shape.m, shape.k];
    let params_buf = ctx.upload_bytes(bytemuck::cast_slice(&params));

    // Switch via env: WICK_Q4_FAST=1 to bench the fast variant.
    let mode = std::env::var("WICK_Q4_FAST").unwrap_or_default();
    if mode == "splitk" {
        return bench_metal_splitk(ctx, &a_buf, &x_buf, shape, iters, expected);
    }
    let (pipeline, groups, threads) = if mode == "slim" {
        (
            ctx.create_pipeline(
                wick::backend::metal::shaders::GEMV_Q4_0_FAST,
                "gemv_q4_0_fast_slim",
            )
            .expect("MSL compile"),
            shape.m.div_ceil(2) as u64,
            32u64,
        )
    } else if mode == "1" {
        (
            ctx.create_pipeline(
                wick::backend::metal::shaders::GEMV_Q4_0_FAST,
                "gemv_q4_0_fast",
            )
            .expect("MSL compile"),
            shape.m.div_ceil(8) as u64,
            64u64,
        )
    } else {
        (
            ctx.create_pipeline(wick::backend::metal::shaders::GEMV_Q4_0, "gemv_q4_0")
                .expect("MSL compile"),
            shape.m.div_ceil(2) as u64,
            32u64,
        )
    };
    let tg_count = MTLSize {
        width: groups,
        height: 1,
        depth: 1,
    };
    let threads_per_tg = MTLSize {
        width: threads,
        height: 1,
        depth: 1,
    };

    let dispatch = |label: Option<&str>| {
        let cb = ctx.queue.new_command_buffer();
        if let Some(l) = label {
            cb.set_label(l);
        }
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&a_buf), 0);
        enc.set_buffer(1, Some(&x_buf), 0);
        enc.set_buffer(2, Some(&y_buf), 0);
        enc.set_buffer(3, Some(&params_buf), 0);
        enc.dispatch_thread_groups(tg_count, threads_per_tg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    };

    // Warmup + correctness check.
    for _ in 0..3 {
        dispatch(Some("warmup"));
    }
    let result = ctx.read_f32(&y_buf, shape.m as usize);
    check_parity("metal", expected, &result);

    // Timed batch — queue all commands, wait on the last one.
    let start = std::time::Instant::now();
    let mut last = None;
    for i in 0..iters {
        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&a_buf), 0);
        enc.set_buffer(1, Some(&x_buf), 0);
        enc.set_buffer(2, Some(&y_buf), 0);
        enc.set_buffer(3, Some(&params_buf), 0);
        enc.dispatch_thread_groups(tg_count, threads_per_tg);
        enc.end_encoding();
        cb.commit();
        if i == iters - 1 {
            last = Some(cb.to_owned());
        }
    }
    if let Some(cb) = last {
        cb.wait_until_completed();
    }
    let elapsed = start.elapsed().as_secs_f64();
    elapsed / iters as f64
}

fn check_parity(label: &str, expected: &[f32], got: &[f32]) {
    assert_eq!(expected.len(), got.len());
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for i in 0..expected.len() {
        let d = (expected[i] - got[i]).abs();
        max_abs = max_abs.max(d);
        let denom = expected[i].abs().max(1e-6);
        max_rel = max_rel.max(d / denom);
    }
    assert!(
        max_abs < 1e-2 || max_rel < 1e-3,
        "{label}: parity mismatch — max_abs={max_abs:.4e}, max_rel={max_rel:.4e}"
    );
    eprintln!("{label}: parity OK (max_abs={max_abs:.2e}, max_rel={max_rel:.2e})");
}

#[test]
#[ignore]
fn bench_q4_0_gemv_wgsl_vs_msl() {
    let shapes = [
        Shape {
            m: 256,
            k: 1024,
            label: "attn-kv",
        },
        Shape {
            m: 1024,
            k: 1024,
            label: "attn-q/out",
        },
        Shape {
            m: 2048,
            k: 1024,
            label: "ffn-up/gate",
        },
        Shape {
            m: 3072,
            k: 1024,
            label: "conv-in-proj",
        },
        Shape {
            m: 4096,
            k: 1024,
            label: "mid-4096",
        },
        Shape {
            m: 1024,
            k: 2048,
            label: "ffn-down",
        },
        Shape {
            m: 65536,
            k: 1024,
            label: "vocab-head",
        },
    ];
    for s in &shapes {
        bench_shape(s, 200);
    }
}

fn bench_metal_splitk(
    ctx: &MetalContext,
    a_buf: &metal::Buffer,
    x_buf: &metal::Buffer,
    shape: &Shape,
    iters: u32,
    expected: &[f32],
) -> f64 {
    use metal::MTLSize;
    let n_splits = 4u32;
    let y_partial = ctx.create_buffer((shape.m as u64) * (n_splits as u64) * 4);
    let y_final = ctx.create_buffer((shape.m as u64) * 4);

    // SplitKParams struct layout: { uint m; uint k; uint n_splits; }
    let params = [shape.m, shape.k, n_splits];

    let pipe_split = ctx
        .create_pipeline(
            wick::backend::metal::shaders::GEMV_Q4_0_FAST,
            "gemv_q4_0_fast_splitk",
        )
        .expect("MSL compile splitk");

    let pipe_merge = ctx
        .create_pipeline(
            wick::backend::metal::shaders::GEMV_Q4_0_FAST,
            "gemv_q4_0_splitk_merge",
        )
        .expect("MSL compile merge");

    let rows_per_split = (shape.m + 7) / 8;
    let grid_split = MTLSize::new((rows_per_split * n_splits) as u64, 1, 1);
    let threads_split = MTLSize::new(64, 1, 1);
    let grid_merge = MTLSize::new(shape.m as u64, 1, 1);
    // One thread per TG: the merge shader does a scalar reduction and writes
    // `y[row]` — with multiple threads per TG all threads race on that write.
    let threads_merge = MTLSize::new(1, 1, 1);

    let dispatch = || {
        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();

        // Phase A
        enc.set_compute_pipeline_state(&pipe_split);
        enc.set_buffer(0, Some(a_buf), 0);
        enc.set_buffer(1, Some(x_buf), 0);
        enc.set_buffer(2, Some(&y_partial), 0);
        enc.set_bytes(3, 12, params.as_ptr() as *const _);
        enc.dispatch_thread_groups(grid_split, threads_split);

        // Phase B
        enc.set_compute_pipeline_state(&pipe_merge);
        enc.set_buffer(0, Some(&y_partial), 0);
        enc.set_buffer(1, Some(&y_final), 0);
        enc.set_bytes(2, 12, params.as_ptr() as *const _);
        enc.dispatch_thread_groups(grid_merge, threads_merge);

        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    };

    for _ in 0..3 {
        dispatch();
    }
    let result = ctx.read_f32(&y_final, shape.m as usize);
    check_parity("metal-splitk", expected, &result);

    let start = std::time::Instant::now();
    for i in 0..iters {
        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe_split);
        enc.set_buffer(0, Some(a_buf), 0);
        enc.set_buffer(1, Some(x_buf), 0);
        enc.set_buffer(2, Some(&y_partial), 0);
        enc.set_bytes(3, 12, params.as_ptr() as *const _);
        enc.dispatch_thread_groups(grid_split, threads_split);
        enc.set_compute_pipeline_state(&pipe_merge);
        enc.set_buffer(0, Some(&y_partial), 0);
        enc.set_buffer(1, Some(&y_final), 0);
        enc.set_bytes(2, 12, params.as_ptr() as *const _);
        enc.dispatch_thread_groups(grid_merge, threads_merge);
        enc.end_encoding();
        cb.commit();
        if i == iters - 1 {
            cb.wait_until_completed();
        }
    }
    start.elapsed().as_secs_f64() / iters as f64
}
