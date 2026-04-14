import os
import subprocess
import re
import csv
import sys
from pathlib import Path

# Paths
HOME = Path.home()
MODELS_DIR = HOME / ".leap" / "models"
WICK_BIN = Path.cwd() / "target" / "release" / "wick"
LLAMA_BENCH = Path("/Users/dberrios/development/llama.cpp/worktrees/dberrios-updateLlama/build/bin/llama-bench")

# Matrix
PROMPT_LENGTHS = [128, 1024, 4096]
GEN_LENGTHS = [64, 256, 1024]
RUNS = 5

def find_models():
    models = []
    if not MODELS_DIR.exists():
        return models
    for root, dirs, files in os.walk(MODELS_DIR):
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

def run_wick(model_path, prompt_len, gen_len):
    print(f"  [wick] p={prompt_len} n={gen_len}...")
    cmd = [
        "/usr/bin/time", "-l",
        str(WICK_BIN), "bench",
        "--model", str(model_path),
        "--prompt-tokens", str(prompt_len),
        "--max-tokens", str(gen_len),
        "--runs", str(RUNS),
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

def run_llama(model_path, prompt_len, gen_len):
    print(f"  [llama] p={prompt_len} n={gen_len}...")
    # llama-bench: -p prompt, -n gen
    cmd = [
        "/usr/bin/time", "-l",
        str(LLAMA_BENCH),
        "-m", str(model_path),
        "-p", str(prompt_len),
        "-n", str(gen_len),
        "-ngl", "99",
        "-r", str(RUNS),
        "--no-warmup" # wick bench has its own warmup, llama-bench warmup can be slow
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # llama-bench outputs a markdown table to stdout.
        # | model | size | params | backend | threads | test | t/s |
        # test is 'pp<prompt>' for prefill, 'tg<gen>' for generation
        
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

def main():
    models = find_models()
    if not models:
        print("No models found in ~/.leap/models")
        return

    output_file = "benchmark_results.csv"
    fields = ["model", "prompt_len", "gen_len", "engine", "prefill_tps", "decode_tps", "rss_mb", "footprint_mb"]
    
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        
        for model in models:
            model_name = model.name
            print(f"\nBenchmarking model: {model_name}")
            for p_len in PROMPT_LENGTHS:
                for g_len in GEN_LENGTHS:
                    # Run Wick
                    p_tps, d_tps, rss, foot = run_wick(model, p_len, g_len)
                    writer.writerow({
                        "model": model_name, "prompt_len": p_len, "gen_len": g_len,
                        "engine": "wick", "prefill_tps": p_tps, "decode_tps": d_tps,
                        "rss_mb": rss / 1024 / 1024, "footprint_mb": foot / 1024 / 1024
                    })
                    
                    # Run Llama.cpp
                    p_tps, d_tps, rss, foot = run_llama(model, p_len, g_len)
                    writer.writerow({
                        "model": model_name, "prompt_len": p_len, "gen_len": g_len,
                        "engine": "llama.cpp", "prefill_tps": p_tps, "decode_tps": d_tps,
                        "rss_mb": rss / 1024 / 1024, "footprint_mb": foot / 1024 / 1024
                    })
                    f.flush()

    print(f"\nDone! Results saved to {output_file}")

if __name__ == "__main__":
    main()
