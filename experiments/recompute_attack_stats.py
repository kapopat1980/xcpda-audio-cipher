"""Re-test the leakage-attack accuracies against pure guessing (1/K) with exact one-sided binomial tests,
Clopper-Pearson intervals and Holm correction over the 18 XC-PDA tests (Section 5.3, Table 9, Figure 6).
Usage: python experiments/recompute_attack_stats.py results/kaggle_run2/partB_dl_attack.json"""
import json, sys
from scipy.stats import binomtest

SHORT = {"environmental-sound-classification-50": "ESC-50", "gtzan-dataset-music-genre-classification": "GTZAN",
         "speech-commands": "Speech Commands"}
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/kaggle_run2/partB_dl_attack.json"))
out, xc = [], []
for ds, r in d.items():
    K, n = r["classes"], r["test_clips"]
    for cond, m in r["conditions"].items():
        for att, v in m.items():
            k = int(round(v["acc"] * n)); bt = binomtest(k, n, 1 / K, alternative="greater"); ci = binomtest(k, n).proportion_ci(0.95)
            row = dict(ds=SHORT.get(ds, ds), cond=cond, att=att, acc=v["acc"], k=k, n=n, K=K, p=bt.pvalue, lo=ci.low, hi=ci.high)
            out.append(row)
            if cond.startswith("xcpda"): xc.append(row)
xc.sort(key=lambda r: r["p"])
for i, r in enumerate(xc):
    r["p_holm"] = min(1.0, r["p"] * (len(xc) - i))
    print(f"{r['ds']:16s} {r['cond']:20s} {r['att']:22s} acc {r['acc']*100:5.1f}%  p={r['p']:.3f}  Holm={r['p_holm']:.3f}")
json.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else "results/kaggle_run2/partB_uniform_baseline.json", "w"), indent=1)
