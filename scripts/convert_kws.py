#!/usr/bin/env python3
"""
Convert or generate Keyword Spotting (KWS) models for Cera.

This script packages trained or reference KWS models into self-describing GGUF
containers with folded BatchNorm layers and comprehensive metadata.
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import gguf
import numpy as np
import torch
import torch.nn as nn


# ── Acoustic Front-End Helpers ────────────────────────────────────────────────


def hz_to_mel(hz: float) -> float:
    """Convert Hz to Mel scale using HTK formula."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    """Convert Mel scale to Hz using HTK formula."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def create_mel_filterbank(
    sr: int = 16000,
    n_fft: int = 512,
    n_mels: int = 32,
    fmin: float = 80.0,
    fmax: float = 7600.0,
) -> np.ndarray:
    """Create triangular Mel filterbank matrix of shape (n_mels, n_fft // 2 + 1)."""
    min_mel = hz_to_mel(fmin)
    max_mel = hz_to_mel(fmax)
    mel_points = np.linspace(min_mel, max_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    n_bins = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)

    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        for k in range(f_m_minus, f_m):
            fb[m - 1, k] = (k - f_m_minus) / max(f_m - f_m_minus, 1)
        for k in range(f_m, f_m_plus):
            fb[m - 1, k] = (f_m_plus - k) / max(f_m_plus - f_m, 1)

    return fb


def extract_log_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = 512,
    win_length: int = 400,
    hop_length: int = 160,
    n_mels: int = 32,
) -> np.ndarray:
    """
    Extract log-mel spectrogram matching Cera's pure-Rust front-end.
    Returns: shape (n_mels, num_frames)
    """
    assert audio.ndim == 1, "audio must be 1D mono PCM"
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(win_length) / win_length)
    fb = create_mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels)

    num_samples = len(audio)
    if num_samples < win_length:
        return np.zeros((n_mels, 0), dtype=np.float32)

    num_frames = (num_samples - win_length) // hop_length + 1
    n_bins = n_fft // 2 + 1
    mel_frames = np.zeros((n_mels, num_frames), dtype=np.float32)

    for i in range(num_frames):
        start = i * hop_length
        frame = audio[start : start + win_length] * window
        padded = np.zeros(n_fft, dtype=np.float32)
        padded[:win_length] = frame

        fft_complex = np.fft.rfft(padded, n=n_fft)
        power_spec = (np.abs(fft_complex) ** 2).astype(np.float32)

        mel_energies = np.dot(fb, power_spec)
        mel_frames[:, i] = np.log1p(np.maximum(mel_energies, 0.0))

    return mel_frames


# ── PyTorch Neural Model Architecture ─────────────────────────────────────────


class ConvBlock(nn.Module):
    """1D Convolution block with BatchNorm and SiLU activation."""

    def __init__(self, in_c: int, out_c: int, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(
            in_c,
            out_c,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_c)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class KwsBackbone(nn.Module):
    """Frozen acoustic embedding backbone mapping log-mel frames to 64-dim embedding."""

    def __init__(self, mel_bins: int = 32, emb_dim: int = 64):
        super().__init__()
        self.conv0 = ConvBlock(mel_bins, 64, stride=2)
        self.conv1 = ConvBlock(64, 64, stride=2)
        self.conv2 = ConvBlock(64, 64, stride=2)
        self.conv3 = ConvBlock(64, emb_dim, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # Temporal mean pooling
        return x.mean(dim=-1)


class KwsHead(nn.Module):
    """Lightweight 2-layer MLP classification head."""

    def __init__(self, emb_dim: int = 64, hidden_dim: int = 32, num_keywords: int = 1):
        super().__init__()
        self.dense1 = nn.Linear(emb_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, num_keywords)
        self.act = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.dense1(emb))
        logits = self.dense2(h)
        return self.sigmoid(logits)


class KwsModel(nn.Module):
    """End-to-end Keyword Spotting model combining backbone and head."""

    def __init__(self, mel_bins: int = 32, emb_dim: int = 64, num_keywords: int = 1):
        super().__init__()
        self.backbone = KwsBackbone(mel_bins=mel_bins, emb_dim=emb_dim)
        self.head = KwsHead(emb_dim=emb_dim, num_keywords=num_keywords)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        emb = self.backbone(mel)
        return self.head(emb)


# ── BatchNorm Folding Optimization ────────────────────────────────────────────


def fold_conv1d_bn(
    conv: nn.Conv1d,
    bn: nn.BatchNorm1d,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Algebraically fold BatchNorm parameters into preceding Conv1d weights and bias.
    W_folded = W * (gamma / sqrt(var + eps))
    b_folded = beta + (b - mean) * (gamma / sqrt(var + eps))
    """
    w = conv.weight.detach().cpu().numpy()
    b = conv.bias.detach().cpu().numpy() if conv.bias is not None else np.zeros(conv.out_channels, dtype=np.float32)

    gamma = bn.weight.detach().cpu().numpy()
    beta = bn.bias.detach().cpu().numpy()
    mean = bn.running_mean.detach().cpu().numpy()
    var = bn.running_var.detach().cpu().numpy()
    eps = bn.eps

    scale = gamma / np.sqrt(var + eps)
    w_folded = w * scale[:, None, None]
    b_folded = beta + (b - mean) * scale

    return w_folded.astype(np.float32), b_folded.astype(np.float32)


# ── GGUF Exporter ─────────────────────────────────────────────────────────────


def export_kws_to_gguf(
    output_path: str,
    keywords: List[str],
    model: KwsModel,
    sample_rate: int = 16000,
    window_samples: int = 19200,
    hop_samples: int = 1280,
    mel_bins: int = 32,
    mel_window_samples: int = 400,
    mel_hop_samples: int = 160,
    fft_size: int = 512,
    embedding_dim: int = 64,
    default_threshold: float = 0.75,
    cooldown_ms: int = 2000,
    pre_roll_ms: int = 150,
) -> None:
    """Export trained KWS model to self-describing GGUF container."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    model.eval()

    writer = gguf.GGUFWriter(output_path, arch="kws")

    # 1. Self-describing GGUF Metadata
    keyword_summary = ", ".join(keywords)
    writer.add_string("general.name", f"KWS Model ({keyword_summary})")
    writer.add_string(
        "general.description",
        f"Native keyword spotting model for Cera detecting: {keyword_summary}",
    )
    writer.add_uint32("kws.keyword_count", len(keywords))
    writer.add_array("kws.keywords", keywords)
    writer.add_uint32("kws.sample_rate", sample_rate)
    writer.add_uint32("kws.window_samples", window_samples)
    writer.add_uint32("kws.hop_samples", hop_samples)
    writer.add_uint32("kws.mel_bins", mel_bins)
    writer.add_uint32("kws.mel_window_samples", mel_window_samples)
    writer.add_uint32("kws.mel_hop_samples", mel_hop_samples)
    writer.add_uint32("kws.fft_size", fft_size)
    writer.add_uint32("kws.embedding_dim", embedding_dim)
    writer.add_float32("kws.default_threshold", default_threshold)
    writer.add_uint32("kws.cooldown_ms", cooldown_ms)
    writer.add_uint32("kws.pre_roll_ms", pre_roll_ms)

    # 2. Extract Folded Conv Tensors
    conv_bn_pairs = [
        ("kws.backbone.conv0", model.backbone.conv0.conv, model.backbone.conv0.bn),
        ("kws.backbone.conv1", model.backbone.conv1.conv, model.backbone.conv1.bn),
        ("kws.backbone.conv2", model.backbone.conv2.conv, model.backbone.conv2.bn),
        ("kws.backbone.conv3", model.backbone.conv3.conv, model.backbone.conv3.bn),
    ]

    tensors_to_write: Dict[str, np.ndarray] = {}

    for prefix, conv, bn in conv_bn_pairs:
        w_folded, b_folded = fold_conv1d_bn(conv, bn)
        tensors_to_write[f"{prefix}.weight"] = w_folded
        tensors_to_write[f"{prefix}.bias"] = b_folded

    # 3. Extract Head Dense Tensors
    dense1_w = model.head.dense1.weight.detach().cpu().numpy().astype(np.float32)
    dense1_b = model.head.dense1.bias.detach().cpu().numpy().astype(np.float32)
    dense2_w = model.head.dense2.weight.detach().cpu().numpy().astype(np.float32)
    dense2_b = model.head.dense2.bias.detach().cpu().numpy().astype(np.float32)

    tensors_to_write["kws.head.dense1.weight"] = dense1_w
    tensors_to_write["kws.head.dense1.bias"] = dense1_b
    tensors_to_write["kws.head.dense2.weight"] = dense2_w
    tensors_to_write["kws.head.dense2.bias"] = dense2_b

    # 4. Write Tensors
    for name, arr in tensors_to_write.items():
        print(f"Adding tensor: {name:32s} shape={str(arr.shape):16s} dtype={arr.dtype}")
        writer.add_tensor(name, arr)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_bytes = os.path.getsize(output_path)
    print(f"\nSuccessfully generated {output_path} ({size_bytes:,} bytes)")


# ── Test Fixture & Verification ───────────────────────────────────────────────


def generate_verification_fixture(
    fixture_path: str,
    model: KwsModel,
    keywords: List[str],
    seed: int = 42,
) -> None:
    """Generate deterministic oracle fixture for cross-language validation."""
    os.makedirs(os.path.dirname(os.path.abspath(fixture_path)), exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model.eval()

    # Generate synthetic 1.2s audio signal with multiple frequencies
    t = np.linspace(0, 1.2, 19200, endpoint=False, dtype=np.float32)
    synthetic_audio = (
        0.3 * np.sin(2.0 * np.pi * 440.0 * t)
        + 0.2 * np.sin(2.0 * np.pi * 880.0 * t)
        + 0.1 * np.sin(2.0 * np.pi * 1760.0 * t)
    ).astype(np.float32)

    # Extract reference mel spectrogram
    mel_frames = extract_log_mel_spectrogram(synthetic_audio)
    mel_tensor = torch.from_numpy(mel_frames).unsqueeze(0)

    # Run reference model forward pass
    with torch.no_grad():
        emb_tensor = model.backbone(mel_tensor)
        probs_tensor = model.head(emb_tensor)

    # Folded conv verification
    conv0_w, conv0_b = fold_conv1d_bn(model.backbone.conv0.conv, model.backbone.conv0.bn)
    c0 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=True)
    c0.weight.data = torch.from_numpy(conv0_w)
    c0.bias.data = torch.from_numpy(conv0_b)
    out_orig = model.backbone.conv0(mel_tensor)
    out_folded = nn.SiLU()(c0(mel_tensor))
    bn_diff = (out_orig - out_folded).abs().max().item()
    assert bn_diff < 1e-5, f"Folded BN difference exceeds tolerance: {bn_diff}"

    fixture_data = {
        "keywords": keywords,
        "sample_rate": 16000,
        "window_samples": 19200,
        "mel_bins": 32,
        "embedding_dim": 64,
        "audio_slice": synthetic_audio[:256].tolist(),
        "mel_slice": mel_frames[:, :8].tolist(),
        "embedding": emb_tensor.squeeze(0).tolist(),
        "probabilities": probs_tensor.squeeze(0).tolist(),
    }

    with open(fixture_path, "w", encoding="utf-8") as f:
        json.dump(fixture_data, f, indent=2)

    print(f"Exported verification oracle fixture to {fixture_path}")


# ── Main Entrypoint ───────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert or generate Keyword Spotting (KWS) models for Cera"
    )
    parser.add_argument(
        "--output",
        default="models/hey_liquid.gguf",
        help="Path to output GGUF model file",
    )
    parser.add_argument(
        "--keywords",
        default="Hey Liquid",
        help="Comma-separated target keywords",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional path to PyTorch checkpoint containing model state_dict",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Default activation threshold (default: 0.75)",
    )
    parser.add_argument(
        "--export-fixture",
        default=None,
        help="Optional path to export JSON oracle fixture for Rust unit tests",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reference weight initialization (default: 42)",
    )
    args = parser.parse_args()

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    if not keywords:
        raise ValueError("At least one target keyword must be specified")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = KwsModel(mel_bins=32, emb_dim=64, num_keywords=len(keywords))

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}...")
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        print("Initializing reference model weights (seed: %d)..." % args.seed)

    export_kws_to_gguf(
        output_path=args.output,
        keywords=keywords,
        model=model,
        default_threshold=args.threshold,
    )

    if args.export_fixture:
        generate_verification_fixture(
            fixture_path=args.export_fixture,
            model=model,
            keywords=keywords,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
