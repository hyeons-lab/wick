import csv

with open('benchmark_results.csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

data = {}
for r in rows:
    key = (r['model'], int(r['prompt_len']), int(r['gen_len']))
    if key not in data:
        data[key] = {}
    data[key][r['engine']] = r

lines = []
lines.append('| Model | Prompt | Gen | Prefill tok/s (Wick / Llama) | Decode tok/s (Wick / Llama) | RSS MB (Wick / Llama) | Footprint MB (Wick / Llama) |')
lines.append('|-------|--------|-----|------------------------------|-----------------------------|-----------------------|-----------------------------|')

for key in sorted(data.keys()):
    engines = data[key]
    if 'wick' not in engines or 'llama.cpp' not in engines:
        continue
    w = engines['wick']
    l = engines['llama.cpp']
    
    w_p = float(w['prefill_tps'])
    l_p = float(l['prefill_tps'])
    if w_p == 0 and l_p == 0: continue
    
    w_d = float(w['decode_tps'])
    l_d = float(l['decode_tps'])
    
    w_rss = float(w['rss_mb'])
    l_rss = float(l['rss_mb'])
    
    w_foot = float(w['footprint_mb'])
    l_foot = float(l['footprint_mb'])

    def format_comp(w_val, l_val, is_speed=True):
        if l_val == 0: return f"{w_val:.1f} / 0.0"
        if w_val == 0: return f"0.0 / {l_val:.1f}"
        ratio = w_val / l_val
        # Format ratio based on context: speedup vs memory usage
        if is_speed:
            if ratio >= 1.0:
                color = "**"
            else:
                color = ""
            return f"{w_val:.1f} / {l_val:.1f} ({color}{ratio:.2f}x{color})"
        else:
            return f"{w_val:.1f} / {l_val:.1f} ({ratio:.2f}x)"

    p_str = format_comp(w_p, l_p, True)
    d_str = format_comp(w_d, l_d, True)
    r_str = format_comp(w_rss, l_rss, False)
    f_str = format_comp(w_foot, l_foot, False)
    
    lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {p_str} | {d_str} | {r_str} | {f_str} |")

print('\n'.join(lines))
