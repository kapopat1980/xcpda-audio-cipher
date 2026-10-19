"""Summarise an NIST STS 2.1.2 finalAnalysisReport.txt per test (multi-instance tests aggregated)."""
import sys, re, json, collections
rows = collections.OrderedDict()
for line in open(sys.argv[1]):
    m = re.match(r"\s*((?:\d+\s+){10})(\S+)\s+\*?\s*(\d+)/\s*(\d+)\s*\*?\s+(\w+)", line)
    if not m:
        continue
    p = m.group(2); p = float(p) if p not in ("----",) else None
    rows.setdefault(m.group(5), []).append((p, int(m.group(3)), int(m.group(4))))
out = {}
for t, v in rows.items():
    props = [a / b for _, a, b in v]
    ps = [p for p, _, _ in v if p is not None]
    out[t] = {"instances": len(v), "min_pass": min(a for _, a, _ in v), "n": v[0][2],
              "min_prop": min(props), "min_uniformity_p": min(ps) if ps else None,
              "uniformity_ok": all(p >= 0.0001 for p in ps)}
json.dump(out, sys.stdout, indent=1)
