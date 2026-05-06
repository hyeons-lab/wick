//! Image decode + resize + normalize for VL input.
//!
//! Takes raw PNG / JPEG bytes and produces a `[3 × H × W]` f32
//! NCHW tensor in the layout
//! [`crate::model::vision_encoder::VisionEncoderWeights::encode_image`]
//! expects: per-channel mean-subtracted then divided by per-channel
//! std, with channel order R, G, B and the inner two dims being
//! the resized image's height × width.
//!
//! Hardcoded per `InferenceType::LlamaCppImageToText` (see
//! `project_no_schema_extensions.md`):
//! - target image_size: from `cfg.image_size` (256 for LFM2.5-VL).
//! - mean / std: from `cfg.image_mean` / `cfg.image_std` (read at
//!   load time from `clip.vision.image_{mean,std}` GGUF metadata).
//! - resize filter: bilinear (`Triangle`) — adequate for the v1
//!   end-to-end smoke; matching `clip.cpp`'s exact filter choice
//!   is part of the deferred parity gate.
//!
//! Gated behind the `vl-preprocess` feature so embedded targets
//! that only do text or raw-PCM audio input can drop the `image`
//! crate dep.

#![cfg(feature = "vl-preprocess")]

use crate::model::vision_encoder::VisionEncoderConfig;
use crate::session::WickError;

/// Decode + resize + normalize an image into the
/// `[3 × image_size × image_size]` f32 NCHW tensor that
/// [`crate::model::vision_encoder::VisionEncoderWeights::encode_image`]
/// consumes. `bytes` may be PNG or JPEG (auto-detected via
/// `image::guess_format`); other formats fall through to a typed
/// `Backend` error from the underlying `image` crate.
pub fn preprocess_image(bytes: &[u8], cfg: &VisionEncoderConfig) -> Result<Vec<f32>, WickError> {
    if bytes.is_empty() {
        return Err(WickError::EmptyInput);
    }

    // Decode → DynamicImage.
    let img = image::load_from_memory(bytes)
        .map_err(|e| WickError::Backend(format!("image decode failed: {e}")))?;

    // Aspect-ratio sanity warning: this preprocessor force-resizes
    // to a square `cfg.image_size × cfg.image_size`. For non-square
    // inputs (especially extreme — a wide banner squashed to 256²
    // loses most of the discriminative signal) the encoder still
    // produces output but a downstream model that depends on
    // dynamic-resolution features (llama.cpp's
    // `mtmd_image_preprocessor_lfm2` resizes within the
    // `[image_min_pixels=65536, image_max_pixels=262144]` band
    // while preserving aspect) won't see the same input.
    // Surface the divergence to logs so a user wondering why their
    // 1024×400 banner produces a vague description sees the cause
    // without reading the devlog. Threshold 1.5 picks up "panorama
    // squashed to square" while letting normal photos through.
    let aspect =
        (img.width() as f32 / img.height() as f32).max(img.height() as f32 / img.width() as f32);
    if aspect >= 1.5 {
        tracing::warn!(
            input_w = img.width(),
            input_h = img.height(),
            target_size = cfg.image_size,
            aspect_ratio = aspect,
            "vision_preprocessor: input aspect ratio >= 1.5 will be \
             force-squashed to a square; encoder output will differ \
             from llama.cpp's dynamic-resolution path. Consider \
             centre-cropping the input to a near-square first.",
        );
    }

    // Resize to image_size × image_size with bilinear (triangle)
    // filter. CLIP-family inputs are square; aspect-preserving
    // resize would mismatch the model's expectations. Convert to
    // `RgbImage` directly in each branch — going through
    // `DynamicImage::ImageRgb8` and a second `.to_rgb8()` would
    // double-copy the buffer for the already-correct-size case.
    let target = cfg.image_size as u32;
    let rgb = if img.width() == target && img.height() == target {
        img.into_rgb8()
    } else {
        image::imageops::resize(
            &img.to_rgb8(),
            target,
            target,
            image::imageops::FilterType::Triangle,
        )
    };

    // Normalize: NCHW f32, `(rgb / 255 - mean) / std` per
    // channel. Channel-first layout `[c, h, w]` so the encoder's
    // `image[c * H * W + h * W + w]` indexing reads correctly.
    let h = rgb.height() as usize;
    let w = rgb.width() as usize;
    debug_assert_eq!(h, cfg.image_size);
    debug_assert_eq!(w, cfg.image_size);
    let mut out = vec![0f32; 3 * h * w];
    let raw = rgb.as_raw(); // [h * w * 3] u8 in row-major HWC
    for c in 0..3 {
        let mean = cfg.image_mean[c];
        let std_inv = 1.0 / cfg.image_std[c];
        for y in 0..h {
            for x in 0..w {
                let src = (y * w + x) * 3 + c;
                let dst = c * h * w + y * w + x;
                let pixel = raw[src] as f32 / 255.0;
                out[dst] = (pixel - mean) * std_inv;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    fn synth_cfg() -> VisionEncoderConfig {
        VisionEncoderConfig {
            n_layer: 12,
            n_embd: 768,
            n_ff: 3072,
            n_head: 12,
            eps: 1e-6,
            image_size: 4,
            patch_size: 2,
            n_patches: 4,
            projection_dim: 1024,
            scale_factor: 2,
            // Pick non-trivial mean / std so a mean/std swap or
            // channel reorder shows up loudly in the assertions.
            image_mean: [0.5, 0.4, 0.3],
            image_std: [0.2, 0.25, 0.5],
        }
    }

    /// Synthesise a 4×4 solid red PNG, run through the
    /// preprocessor, and assert per-channel normalisation lands
    /// where expected. Red = (1.0, 0.0, 0.0) post-÷255, so:
    ///   R: (1.0 - 0.5) / 0.2  =  2.5
    ///   G: (0.0 - 0.4) / 0.25 = -1.6
    ///   B: (0.0 - 0.3) / 0.5  = -0.6
    /// Catches mean/std ordering bugs (channel mix-up) and
    /// per-channel std=0 div-by-zero (cfg sanity also catches
    /// that, but a 0 in std would explode here too).
    #[test]
    fn preprocess_solid_red_normalises_per_channel() {
        let cfg = synth_cfg();
        let img = ImageBuffer::<Rgb<u8>, _>::from_fn(4, 4, |_, _| Rgb([255u8, 0, 0]));
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .expect("encode test png");

        let out = preprocess_image(&bytes, &cfg).expect("preprocess");
        assert_eq!(out.len(), 3 * 4 * 4);
        let n = 4 * 4;
        // R-channel block: indices [0, n).
        for &v in &out[0..n] {
            assert!((v - 2.5).abs() < 1e-5, "R channel: {v}");
        }
        // G-channel block: indices [n, 2n).
        for &v in &out[n..2 * n] {
            assert!((v - (-1.6)).abs() < 1e-5, "G channel: {v}");
        }
        // B-channel block: indices [2n, 3n).
        for &v in &out[2 * n..3 * n] {
            assert!((v - (-0.6)).abs() < 1e-5, "B channel: {v}");
        }
    }

    /// `EmptyInput` on zero-byte input — gates the empty case
    /// before we hand off to `image::load_from_memory` (which
    /// would surface a less actionable "unable to determine
    /// format" error).
    #[test]
    fn preprocess_empty_bytes_errors() {
        let cfg = synth_cfg();
        match preprocess_image(&[], &cfg) {
            Err(WickError::EmptyInput) => {}
            other => panic!("expected EmptyInput, got {other:?}"),
        }
    }

    /// Resize path: small JPEG (8×8) → resized to cfg.image_size
    /// (4×4). Verifies the auto-detect dispatch + the resize
    /// branch fires when input dims don't match.
    #[test]
    fn preprocess_jpeg_resizes_to_target() {
        let cfg = synth_cfg();
        let img = ImageBuffer::<Rgb<u8>, _>::from_fn(8, 8, |_, _| Rgb([255u8, 0, 0]));
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Jpeg,
            )
            .expect("encode test jpeg");

        let out = preprocess_image(&bytes, &cfg).expect("preprocess");
        assert_eq!(out.len(), 3 * 4 * 4);
        // JPEG is lossy — assert the R channel lands roughly
        // where the lossless test does (allow 0.1 delta to
        // absorb JPEG quantisation).
        let n = 4 * 4;
        let r_avg = out[0..n].iter().sum::<f32>() / (n as f32);
        assert!((r_avg - 2.5).abs() < 0.1, "R channel mean: {r_avg}");
    }
}
