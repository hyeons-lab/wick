import os
import subprocess
import re
import csv
import sys
import argparse
from pathlib import Path

def find_models(models_dir):
    models = []
    if not models_dir.exists():
        return models
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            # Filter for LLM/VL models (wick only supports LFM2 architecture currently)
            if file.endswith(".gguf") and "LFM2" in file and "vocoder" not in file and "Audio" not in file:
                models.append(Path(root) / file)
    return sorted(models)

def parse_time_output(stderr):
    """Parse /usr/bin/time -l output for RSS and peak memory footprint."""
    rss = 0
    footprint = 0
    for line in stderr.splitlines():
        if "maximum resident set size" in line:
            rss = int(line.strip().split()[0])
        if "peak memory footprint" in line:
            footprint = int(line.strip().split()[0])
    return rss, footprint

def run_wick(wick_bin, model_path, prompt_len, gen_len, runs):
    print(f"  [wick] p={prompt_len} n={gen_len}...")
    cmd = [
        "/usr/bin/time", "-l",
        str(wick_bin), "bench",
        "--model", str(model_path),
        "--prompt-tokens", str(prompt_len),
        "--max-tokens", str(gen_len),
        "--runs", str(runs),
        "--device", "metal",
        "--context-size", "16384",
        "--no-cache"
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Parse tok/s from stderr (wick bench outputs diagnostics to stderr)
        decode_match = re.search(r"decode tok/s: p50=([\d.]+)", proc.stderr)
        prefill_match = re.search(r"prefill tok/s: p50=([\d.]+)", proc.stderr)
        
        decode_tps = float(decode_match.group(1)) if decode_match else 0.0
        prefill_tps = float(prefill_match.group(1)) if prefill_match else 0.0
        
        rss, footprint = parse_time_output(proc.stderr)
        return prefill_tps, decode_tps, rss, footprint
    except Exception as e:
        print(f"    Error running wick: {e}")
        return 0.0, 0.0, 0, 0

def run_llama(llama_bench, model_path, prompt_len, gen_len, runs):
    print(f"  [llama] p={prompt_len} n={gen_len}...")
    # llama-bench: -p prompt, -n gen
    cmd = [
        "/usr/bin/time", "-l",
        str(llama_bench),
        "-m", str(model_path),
        "-p", str(prompt_len),
        "-n", str(gen_len),
        "-ngl", "99",
        "-r", str(runs),
        "--no-warmup"
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # llama-bench outputs a markdown table to stdout.
        
        prefill_tps = 0.0
        decode_tps = 0.0
        
        for line in proc.stdout.splitlines():
            if f"pp{prompt_len}" in line:
                m = re.search(r"([\d.]+)\s±", line)
                if m: prefill_tps = float(m.group(1))
            if f"tg{gen_len}" in line:
                m = re.search(r"([\d.]+)\s±", line)
                if m: decode_tps = float(m.group(1))
        
        rss, footprint = parse_time_output(proc.stderr)
        return prefill_tps, decode_tps, rss, footprint
    except Exception as e:
        print(f"    Error running llama: {e}")
        return 0.0, 0.0, 0, 0

def resolve_llama_bench():
    import shutil
    env_value = os.environ.get("LLAMA_BENCH")
    if env_value:
        return Path(env_value).expanduser()

    resolved = shutil.which("llama-bench")
    if resolved:
        return Path(resolved)

    return Path("llama-bench")

def main():
    parser = argparse.ArgumentParser(description="Wick vs llama.cpp benchmark matrix")
    parser.add_argument("--models-dir", type=Path, default=Path.home() / ".leap" / "models", help="Directory containing GGUF models")
    parser.add_argument("--wick-bin", type=Path, default=Path.cwd() / "target" / "release" / "wick", help="Path to wick binary")
    parser.add_argument("--llama-bench", type=Path, default=resolve_llama_bench(), help="Path to llama-bench binary")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.csv"), help="Output CSV file")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per configuration")
    
    args = parser.parse_args()

    PROMPT_LENGTHS = [128, 1024, 4096]
    GEN_LENGTHS = [64, 256, 1024]

    models = find_models(args.models_dir)
    if not models:
        print(f"No compatible models found in {args.models_dir}")
        return

    fields = ["model", "prompt_len", "gen_len", "engine", "prefill_tps", "decode_tps", "rss_mb", "footprint_mb"]
    
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        
        for model in models:
            model_name = model.name
            print(f"\nBenchmarking model: {model_name}")
            for p_len in PROMPT_LENGTHS:
                for g_len in GEN_LENGTHS:
                    # Run Wick
                    p_tps, d_tps, rss, foot = run_wick(args.wick_bin, model, p_len, g_len, args.runs)
                    writer.writerow({
                        "model": model_name, "prompt_len": p_len, "gen_len": g_len,
                        "engine": "wick", "prefill_tps": p_tps, "decode_tps": d_tps,
                        "rss_mb": rss / 1024 / 1024, "footprint_mb": foot / 1024 / 1024
                    })
                    
                    # Run Llama.cpp
                    p_tps, d_tps, rss, foot = run_llama(args.llama_bench, model, p_len, g_len, args.runs)
                    writer.writerow({
                        "model": model_name, "prompt_len": p_len, "gen_len": g_len,
                        "engine": "llama.cpp", "prefill_tps": p_tps, "decode_tps": d_tps,
                        "rss_mb": rss / 1024 / 1024, "footprint_mb": foot / 1024 / 1024
                    })
                    f.flush()

    print(f"\nDone! Results saved to {args.output}")

if __name__ == "__main__":
    main()
