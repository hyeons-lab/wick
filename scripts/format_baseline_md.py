#!/usr/bin/env python3
"""
Format bench_results.csv into the markdown baseline table the plan expects.

Output shape (one block per model):

    ### LFM2-VL-450M Q4_0 (canonical: --runs 20 --warmup 3, p=4096 at --runs 10)

    | prompt | metal prefill | gpu prefill | cpu prefill | metal tg128 | gpu tg128 | cpu tg128 |
    | -----: | ------------: | ----------: | ----------: | ----------: | --------: | --------: |
    |   p128 |          5711 |         ... |         ... |       396.3 |       ... |       ... |
    ...
"""
from __future__ import annotations
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROW_ORDER = ["128", "1024", "4096"]
COL_ORDER = ["metal", "gpu", "cpu"]


def fmt(v: str) -> str:
    if v == "FAIL" or v == "":
        return "  —  "
    f = float(v)
    return f"{f:.1f}" if f < 1000 else f"{f:.0f}"


def main(csv_path: str = "bench_results.csv") -> None:
    rows = list(csv.DictReader(Path(csv_path).read_text().splitlines()))
    by_model: dict[str, dict[tuple[str, str], dict[str, str]]] = defaultdict(dict)
    for r in rows:
        by_model[r["model"]][(r["device"], r["prompt_tokens"])] = r

    for model in sorted(by_model.keys()):
        cells = by_model[model]
        full = "LFM2-VL-450M" if model == "450M" else "LFM2.5-VL-1.6B"
        print(f"### {full} Q4_0 (canonical: --runs 20 --warmup 3, p=4096 cpu/gpu at --runs 10)")
        print()
        head_l = "| prompt | " + " | ".join(f"{d} prefill" for d in COL_ORDER) + " | " + " | ".join(f"{d} tg128" for d in COL_ORDER) + " |"
        sep_l = "| -----: | " + " | ".join("------------:" for _ in COL_ORDER) + " | " + " | ".join("----------:" for _ in COL_ORDER) + " |"
        print(head_l)
        print(sep_l)
        for ptok in ROW_ORDER:
            cols_pre = []
            cols_dec = []
            for dev in COL_ORDER:
                cell = cells.get((dev, ptok))
                if cell is None:
                    cols_pre.append("  —  ")
                    cols_dec.append("  —  ")
                else:
                    cols_pre.append(fmt(cell["prefill_p50"]))
                    cols_dec.append(fmt(cell["decode_p50"]))
            print(f"|  p{ptok:>4} | " + " | ".join(f"{v:>13}" for v in cols_pre) + " | " + " | ".join(f"{v:>9}" for v in cols_dec) + " |")
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bench_results.csv")
