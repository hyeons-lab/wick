import csv
with open('benchmark_results.csv') as f:
    reader = csv.DictReader(f)
    rows = [r for r in reader if not (float(r['prefill_tps']) == 0 and float(r['decode_tps']) == 0)]

lines = []
lines.append('| Model | Prompt | Gen | Engine | Prefill tok/s | Decode tok/s | RSS (MB) | Footprint (MB) |')
lines.append('|-------|--------|-----|--------|---------------|--------------|----------|----------------|')

for r in rows:
    lines.append(f"| {r['model']} | {r['prompt_len']} | {r['gen_len']} | {r['engine']} | {float(r['prefill_tps']):.1f} | {float(r['decode_tps']):.1f} | {float(r['rss_mb']):.1f} | {float(r['footprint_mb']):.1f} |")

with open('results_table.md', 'w') as f:
    f.write('\n'.join(lines))
