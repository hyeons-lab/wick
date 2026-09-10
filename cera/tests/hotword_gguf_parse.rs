#![cfg(all(feature = "mmap", not(target_arch = "wasm32")))]

use std::path::PathBuf;

use anyhow::Result;
use cera::gguf::GgufFile;
use cera::tensor::DType;

fn find_hotword_model() -> Option<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join("models/hey_liquid.gguf"),
        manifest_dir.join("../models/hey_liquid.gguf"),
        PathBuf::from("models/hey_liquid.gguf"),
    ];
    candidates.into_iter().find(|p| p.exists())
}

#[test]
fn test_hotword_gguf_metadata_and_tensors() -> Result<()> {
    let Some(model_path) = find_hotword_model() else {
        eprintln!("Skipping test: models/hey_liquid.gguf not found");
        return Ok(());
    };

    let gguf = GgufFile::open(&model_path)?;

    // 1. Verify general architecture and descriptive metadata
    assert_eq!(gguf.architecture(), Some("kws"));
    assert!(
        gguf.get_str("general.name")
            .unwrap_or_default()
            .contains("KWS Model")
    );

    // 2. Verify self-describing KWS metadata
    assert_eq!(gguf.get_u32("kws.keyword_count"), Some(1));
    let keywords = gguf
        .get_string_array("kws.keywords")
        .expect("missing kws.keywords array");
    assert_eq!(keywords, vec!["Hey Liquid"]);

    assert_eq!(gguf.get_u32("kws.sample_rate"), Some(16000));
    assert_eq!(gguf.get_u32("kws.window_samples"), Some(19200));
    assert_eq!(gguf.get_u32("kws.hop_samples"), Some(1280));
    assert_eq!(gguf.get_u32("kws.mel_bins"), Some(32));
    assert_eq!(gguf.get_u32("kws.mel_window_samples"), Some(400));
    assert_eq!(gguf.get_u32("kws.mel_hop_samples"), Some(160));
    assert_eq!(gguf.get_u32("kws.fft_size"), Some(512));
    assert_eq!(gguf.get_u32("kws.embedding_dim"), Some(64));
    assert_eq!(gguf.get_f32("kws.default_threshold"), Some(0.75));
    assert_eq!(gguf.get_u32("kws.cooldown_ms"), Some(2000));
    assert_eq!(gguf.get_u32("kws.pre_roll_ms"), Some(150));

    // 3. Verify folded convolutional backbone tensors
    let conv_checks = [
        ("kws.backbone.conv0.weight", 64 * 32 * 3),
        ("kws.backbone.conv0.bias", 64),
        ("kws.backbone.conv1.weight", 64 * 64 * 3),
        ("kws.backbone.conv1.bias", 64),
        ("kws.backbone.conv2.weight", 64 * 64 * 3),
        ("kws.backbone.conv2.bias", 64),
        ("kws.backbone.conv3.weight", 64 * 64 * 3),
        ("kws.backbone.conv3.bias", 64),
        ("kws.head.dense1.weight", 32 * 64),
        ("kws.head.dense1.bias", 32),
        ("kws.head.dense2.weight", 32),
        ("kws.head.dense2.bias", 1),
    ];

    for (tensor_name, expected_numel) in conv_checks {
        let tensor = gguf
            .get_tensor(tensor_name)
            .unwrap_or_else(|e| panic!("missing tensor '{tensor_name}': {e}"));
        assert_eq!(
            tensor.dtype(),
            DType::F32,
            "tensor '{tensor_name}' expected F32 dtype"
        );
        assert_eq!(
            tensor.numel(),
            expected_numel,
            "tensor '{tensor_name}' expected {expected_numel} elements, got {}",
            tensor.numel()
        );
        assert!(
            tensor.as_f32_slice().iter().all(|x| x.is_finite()),
            "tensor '{tensor_name}' contains non-finite values"
        );
    }

    Ok(())
}
