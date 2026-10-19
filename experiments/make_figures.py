"""Render all manuscript figures at 300 dpi.
Usage, from the repository root, after experiments/run_local_experiments.py has written results/local/audio:
    python experiments/make_figures.py [figures] [results/local/audio] [results/kaggle_run2/partB_uniform_baseline.json]"""
import sys, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.io import wavfile
OUT = sys.argv[1] if len(sys.argv) > 1 else "figures"
AUD = sys.argv[2] if len(sys.argv) > 2 else "results/local/audio"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888", "axes.labelcolor": "#222", "xtick.color": "#444", "ytick.color": "#444",
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
GREY, ACC, LIGHT = "#6b7280", "#2563eb", "#c7ccd4"

def load(n):
    sr, o = wavfile.read(f"{AUD}/{n}_orig.wav"); _, e = wavfile.read(f"{AUD}/{n}_enc.wav")
    if o.ndim > 1: o, e = o[:, 0], e[:, 0]
    return sr, o.astype(float), e.astype(float)

# ---- Figure 1: architecture -------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 3.5)); ax.set_xlim(0, 760); ax.set_ylim(0, 350); ax.axis("off")
def box(x, y, w, h, title, lines=(), main=False, tsize=8.5, weight="bold"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8", lw=2 if main else 1.0,
                                ec=ACC if main else "#8a8f98", fc=(0.145, 0.388, 0.922, 0.10) if main else "white"))
    ax.text(x + w / 2, y + h - 18 if lines else y + h / 2, title, ha="center", va="center", fontsize=tsize, fontweight=weight, color="#111")
    for i, l in enumerate(lines): ax.text(x + w / 2, y + h - 40 - 15 * i, l, ha="center", va="center", fontsize=7.3, color="#333")
def arrow(x1, y1, x2, y2, label=None, lx=0, ly=0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=1.0, color="#8a8f98", shrinkA=0, shrinkB=0))
    if label: ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontsize=7.3, color="#555", va="center")
box(10, 272, 140, 58, "Master key K", ["256 bits"])
ax.text(310 + 70, 108, "new in this work", ha="center", fontsize=7, color=ACC, style="italic")
box(180, 272, 570, 58, "Stage 0: HKDF-SHA256", ["salt = nonce n (192 bits);  info = version || header H || frame index k || final flag"])
arrow(150, 301, 180, 301)
xs, bw, yb, bh = [10, 160, 310, 460, 610], 140, 120, 96
titles = ["PCM frame", "1. Permute", "2. Diffuse", "3. Mask", "4. Tag"]
bodies = [["N samples,", "16-bit, mono", "or interleaved"], ["Fisher-Yates", "shuffle keyed", "by ChaCha20"],
          ["forward pass,", "backward pass,", "keyed add-rotate"], ["XOR with", "ChaCha20", "keystream"], ["Poly1305 over", "H, n, k, flag", "and C"]]
for x, t, ls, m in zip(xs, titles, bodies, [False, False, True, False, False]): box(x, yb, bw, bh, t, ls, m)
for i in range(4): arrow(xs[i] + bw, yb + bh / 2, xs[i + 1], yb + bh / 2)
for x, lab in zip(xs[1:], ["K_perm", "K_diff", "K_mask", "K_mac"]): arrow(x + bw / 2, 272, x + bw / 2, yb + bh, lab, 6, 0)
box(460, 20, 290, 48, "Output: H, n, then (C_k, T_k) per frame", tsize=7.5, weight="normal")
arrow(460 + bw / 2, yb, 460 + bw / 2, 68); arrow(610 + bw / 2, yb, 610 + bw / 2, 68)
ax.text(14, 44, "Decryption verifies T_k first, then removes the\nmask, the diffusion and the permutation.", fontsize=7.3, color="#555", va="center")
fig.savefig(f"{OUT}/fig1_architecture.png"); plt.close()

# ---- Figure 2: waveforms ----------------------------------------------------------
fig, ax = plt.subplots(3, 2, figsize=(7.2, 5.0), sharex="col")
for c, (n, t) in enumerate([("romans", "romans.wav, 11.025 kHz mono"), ("synthetic", "synthetic stereo music, 44.1 kHz (left)")]):
    sr, o, e = load(n); tt = np.arange(o.size) / sr
    for r, (y, lab, col) in enumerate([(o, "(a) Original" if c == 0 else "(d) Original", GREY), (e, "(b) Encrypted" if c == 0 else "(e) Encrypted", ACC), (o, "(c) Decrypted (bit-exact)" if c == 0 else "(f) Decrypted (bit-exact)", GREY)]):
        ax[r, c].plot(tt, y / 32768, color=col, lw=0.25, rasterized=True); ax[r, c].set_ylim(-1.05, 1.05)
        ax[r, c].set_title(lab + (f" - {t}" if r == 0 else ""), fontsize=8.5, loc="left")
    ax[2, c].set_xlabel("Time (s)")
for a in ax[:, 0]: a.set_ylabel("Amplitude")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_waveforms.png"); plt.close()

# ---- Figure 3: spectrograms ------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(7.2, 4.6))
for c, (n, t) in enumerate([("romans", "romans.wav"), ("synthetic", "synthetic music (left)")]):
    sr, o, e = load(n)
    for r, (y, lab) in enumerate([(o, "Original"), (e, "Encrypted")]):
        ax[r, c].specgram(y, NFFT=512, Fs=sr, noverlap=256, cmap="Greys", vmin=-40, vmax=60, rasterized=True)
        ax[r, c].set_title(f"({'abcd'[2 * c + r]}) {lab} - {t}", fontsize=8.5, loc="left")
        ax[r, c].set_ylabel("Frequency (Hz)"); ax[r, c].set_xlabel("Time (s)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_spectrograms.png"); plt.close()

# ---- Figure 4: histogram + scatter ------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.0))
sr, o, e = load("drama")
ax[0, 0].hist(o, bins=256, color=GREY); ax[0, 0].set_title("(a) Histogram, original drama.wav", fontsize=8.5, loc="left")
ax[0, 1].hist(e, bins=256, color=ACC); ax[0, 1].set_title("(b) Histogram, encrypted", fontsize=8.5, loc="left")
for a in ax[0]: a.set_xlabel("Sample value"); a.set_ylabel("Count")
sr, o, e = load("synthetic"); k = np.random.default_rng(0).choice(o.size - 1, 6000, replace=False)
ax[1, 0].scatter(o[k], o[k + 1], s=0.8, color=GREY, rasterized=True); ax[1, 0].set_title("(c) Adjacent samples, original music (r = 0.996)", fontsize=8.5, loc="left")
ax[1, 1].scatter(e[k], e[k + 1], s=0.8, color=ACC, rasterized=True); ax[1, 1].set_title("(d) Adjacent samples, encrypted (r = -0.0006)", fontsize=8.5, loc="left")
for a in ax[1]: a.set_xlabel("x(n)"); a.set_ylabel("x(n+1)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_hist_scatter.png"); plt.close()

# ---- Figure 5: throughput -------------------------------------------------------------
rows = [("AES-256-GCM (AEAD)", 8480, 0), ("AES-256-CTR", 4854, 0), ("ChaCha20", 4265, 0), ("ChaCha20-Poly1305 (AEAD)", 2616, 0),
        ("XC-PDA, 1 s frames", 92.7, 1), ("XC-PDA decrypt, whole file", 30.1, 1), ("XC-PDA, whole file", 25.1, 1)]
rt = 0.1764
fig, ax = plt.subplots(figsize=(7.2, 3.0))
for i, (n, v, m) in enumerate(rows):
    y = len(rows) - 1 - i; c = ACC if m else "#9aa0a8"
    ax.hlines(y, 0.1, v, color=c, alpha=0.35, lw=2); ax.plot(v, y, "o", color=c, ms=7)
    ax.text(v * 1.25, y, f"{v:,.1f} MB/s, {v / rt:,.0f}x real time" if m else f"{v:,.0f} MB/s", va="center", fontsize=8, color="#222" if m else "#555")
ax.axvline(rt, color="#555", ls="--", lw=1); ax.text(rt * 1.1, len(rows) - 0.35, "real-time need, 44.1 kHz 16-bit stereo (0.176 MB/s)", fontsize=7.5, color="#555")
ax.set_xscale("log"); ax.set_xlim(0.1, 60000); ax.set_xticks([0.1, 1, 10, 100, 1000, 10000]); ax.set_xticklabels(["0.1", "1", "10", "100", "1,000", "10,000"]); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1])
for t, (n, v, m) in zip(ax.get_yticklabels(), rows[::-1]): t.set_fontweight("bold" if m else "normal")
ax.set_xlabel("Encryption throughput (MB/s, log scale); 60 s of 44.1 kHz stereo, one 2.1 GHz Xeon core")
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0); ax.set_ylim(-0.6, len(rows) - 0.1)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_throughput.png"); plt.close()

# ---- Figure 6: leakage attack -------------------------------------------------------
data = json.load(open(sys.argv[3] if len(sys.argv) > 3 else "results/kaggle_run2/partB_uniform_baseline.json"))
lab = {"plaintext": "Plaintext", "permutation_only_fixed_key": "Permutation only", "chacha20_nonce_reused": "ChaCha20, nonce reused",
       "xcpda_nonce_reused": "XC-PDA, nonce reused", "xcpda_random_nonce": "XC-PDA, normal use"}
best = {}
for r in data:
    k = (r["ds"], r["cond"])
    if k not in best or r["acc"] > best[k]["acc"]: best[k] = r
dsets = ["Speech Commands", "ESC-50", "GTZAN"]
fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.2), sharex=True)
for ax, ds in zip(axes, dsets):
    for i, c in enumerate(lab):
        r = best[(ds, c)]; p0 = 1 / r["K"]; y = 4 - i; m = c.startswith("xcpda"); col = ACC if m else "#9aa0a8"
        ax.hlines(y, r["lo"] / p0, r["hi"] / p0, color=col, alpha=0.5, lw=2); ax.plot(r["acc"] / p0, y, "o", color=col, ms=7)
        txt = (f"p = {r['p']:.2g}" if m else f"{r['acc'] * 100:.1f} %")
        ax.text(r["hi"] / p0 * 1.08, y, txt, va="center", fontsize=7.8, color="#444")
    ax.axvline(1, color="#555", ls="--", lw=1)
    ax.set_yticks(range(5)); ax.set_yticklabels(list(lab.values())[::-1])
    for t in ax.get_yticklabels(): t.set_fontweight("bold" if t.get_text().startswith("XC-PDA") else "normal")
    ax.set_title(f"{ds}  (K = {best[(ds, 'plaintext')]['K']} classes, {best[(ds, 'plaintext')]['n']} test clips)", fontsize=9, loc="left", fontweight="bold")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0); ax.set_ylim(-0.6, 4.6)
axes[0].text(1.06, 4.45, "pure guessing (1/K)", fontsize=7.5, color="#555")
axes[-1].set_xscale("log"); axes[-1].set_xlim(0.4, 45)
axes[-1].set_xticks([0.5, 1, 2, 5, 10, 20]); axes[-1].set_xticklabels(["0.5x", "1x", "2x", "5x", "10x", "20x"])
axes[-1].set_xlabel("Best attacker's test accuracy / pure-guessing accuracy (log scale); whiskers: 95 % Clopper-Pearson intervals")
fig.tight_layout(); fig.savefig(f"{OUT}/fig6_leakage_attack.png"); plt.close()
print("figures written to", OUT)
