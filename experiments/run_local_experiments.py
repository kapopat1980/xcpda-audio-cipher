"""Reproduce every local result in the paper (Tables 2-6, the ablation, nonce reuse, tamper detection and
throughput). Usage, from the repository root:
    python experiments/run_local_experiments.py [--gallery /usr/lib/libreoffice/share/gallery/sounds] [--out results/local]
The eight recorded clips are the LibreOffice gallery sounds (installed with LibreOffice); the two synthetic
signals are generated below with fixed seeds."""
import argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_ap = argparse.ArgumentParser()
_ap.add_argument("--gallery", default=os.environ.get("XC_GALLERY", "/usr/lib/libreoffice/share/gallery/sounds"))
_ap.add_argument("--out", default="results/local")
_ap.add_argument("--audio", default="results/local/audio")
_args = _ap.parse_args()
import os, json, time, struct, platform
import numpy as np
from scipy.io import wavfile
from scipy.signal import welch
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305, AESGCM
from xcpda import encrypt_audio, decrypt_audio, cipher_samples, AuthError, encrypt_chunk, make_header
import nist

OUT = _args.out; AUD = _args.audio; os.makedirs(OUT, exist_ok=True); os.makedirs(AUD, exist_ok=True)
rng = np.random.default_rng(2026)
GAL = _args.gallery.rstrip("/") + "/"

# ---------------- test corpus ----------------
def synth_music_stereo(sec, sr=44100):
    t = np.arange(int(sec * sr)) / sr
    chords = [[261.63, 329.63, 392.0], [220.0, 261.63, 329.63], [174.61, 220.0, 261.63], [196.0, 246.94, 293.66]]
    L = np.zeros_like(t); R = np.zeros_like(t)
    beat = 0.5
    for k in range(int(sec / beat)):
        s = int(k * beat * sr); e = min(int((k + 1) * beat * sr) + sr // 4, t.size)
        tt = t[s:e] - t[s]
        env = np.exp(-3.0 * tt)
        for j, f in enumerate(chords[(k // 4) % 4]):
            note = sum(np.sin(2 * np.pi * f * h * tt) / h ** 1.3 for h in range(1, 7)) * env
            pan = 0.3 + 0.2 * j
            L[s:e] += (1 - pan) * note; R[s:e] += pan * note
    L += 0.02 * rng.standard_normal(t.size); R += 0.02 * rng.standard_normal(t.size)
    m = max(np.abs(L).max(), np.abs(R).max())
    st = np.stack([L, R], axis=1) / m * 0.85 * 32767
    return st.astype(np.int16)

def synth_speech(sec, sr=44100):
    t = np.arange(int(sec * sr)) / sr
    f0 = 120 + 25 * np.sin(2 * np.pi * 0.7 * t) + 10 * np.sin(2 * np.pi * 3.1 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    src = sum(np.sin(h * phase) / h for h in range(1, 40))
    vowels = [(730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240), (530, 1840, 2480), (660, 1720, 2410)]
    out = np.zeros_like(t); seg = int(0.18 * sr)
    from scipy.signal import lfilter
    for k in range(t.size // seg):
        s, e = k * seg, (k + 1) * seg
        if k % 6 == 5: continue  # pause
        y = src[s:e]
        for F, bw in zip(vowels[k % 5], (80, 100, 120)):
            r = np.exp(-np.pi * bw / sr); th = 2 * np.pi * F / sr
            y = lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], y)
        out[s:e] = y * np.hanning(e - s)
    out += 0.003 * rng.standard_normal(t.size) * out.std()
    return (out / np.abs(out).max() * 0.8 * 32767).astype(np.int16)

corpus = {}
for name, f in [("Applause", "applause"), ("Drama (music)", "drama"), ("Gong", "gong"), ("Nature", "nature1"),
                ("Kongas (percussion)", "kongas"), ("Romans (voice)", "romans"), ("Train", "train"), ("Untie (effect)", "untie")]:
    sr, x = wavfile.read(GAL + f + ".wav"); corpus[name] = (sr, 1, x.astype(np.int16))
corpus["Synthetic speech"] = (44100, 1, synth_speech(5))
mus = synth_music_stereo(10)
corpus["Synthetic stereo music"] = (44100, 2, mus.reshape(-1))
for name, (sr, ch, x) in corpus.items():
    wavfile.write(f"{AUD}/{name.split()[0].lower()}_orig.wav", sr, x.reshape(-1, ch) if ch > 1 else x)

# ---------------- metrics ----------------
def entropy(v, bits):
    c = np.bincount(v, minlength=1 << bits).astype(float); p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum())

def to_u16(x): return (x.astype(np.int64) & 0xFFFF)
def to_s(x): return ((x.astype(np.int64) + 32768) & 0xFFFF) - 32768

def adj_corr(x, ch):
    x = x.reshape(-1, ch)[:, 0].astype(float)
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])

def chi_bytes(u16):
    b = u16.astype("<u2").tobytes(); c = np.bincount(np.frombuffer(b, np.uint8), minlength=256)
    e = c.sum() / 256; return float(((c - e) ** 2 / e).sum())

def snr(orig, other):
    o = orig.astype(float); d = o - other.astype(float)
    return float(10 * np.log10((o ** 2).sum() / (d ** 2).sum()))

def flatness(x, sr):
    f, P = welch(x.astype(float), sr, nperseg=1024); P = P[1:]
    return float(np.exp(np.mean(np.log(P))) / np.mean(P))

def nscr_uaci(c1, c2):
    d = np.abs(c1.astype(np.int64) - c2.astype(np.int64))
    return float(np.mean(d != 0) * 100), float(np.mean(d) / 65535 * 100)

def stream_cipher(kind, key, nonce, x):
    data = to_u16(x).astype("<u2").tobytes()
    if kind == "chacha20":
        e = Cipher(algorithms.ChaCha20(key, nonce[:16]), None).encryptor()
    else:
        e = Cipher(algorithms.AES(key), modes.CTR(nonce[:16])).encryptor()
    return np.frombuffer(e.update(data), "<u2").astype(np.int64)

KEY = bytes(range(32)); NONCE = bytes(range(100, 124))
TRIALS = 20
res = {"files": {}, "platform": platform.processor() or platform.machine()}

for name, (sr, ch, x) in corpus.items():
    blob = encrypt_audio(KEY, x, sr, ch, nonce=NONCE)
    c = cipher_samples(blob)
    y = decrypt_audio(KEY, blob)
    assert np.array_equal(x, y)
    cs = to_s(c)
    wavfile.write(f"{AUD}/{name.split()[0].lower()}_enc.wav", sr, cs.astype(np.int16).reshape(-1, ch) if ch > 1 else cs.astype(np.int16))
    ideal = float(np.mean([entropy(rng.integers(0, 65536, x.size), 16) for _ in range(5)]))
    r = {"H16_ideal": ideal, "sr": sr, "ch": ch, "samples": int(x.size), "seconds": round(x.size / ch / sr, 2),
         "H16_orig": entropy(to_u16(x), 16), "H16_enc": entropy(c, 16),
         "H8_orig": entropy(np.frombuffer(to_u16(x).astype('<u2').tobytes(), np.uint8), 8),
         "H8_enc": entropy(np.frombuffer(c.astype('<u2').tobytes(), np.uint8), 8),
         "corr_orig": adj_corr(x, ch), "corr_enc": adj_corr(cs, ch),
         "chi2_orig": chi_bytes(to_u16(x)), "chi2_enc": chi_bytes(c),
         "snr_enc": snr(x, cs), "flat_orig": flatness(x.reshape(-1, ch)[:, 0], sr), "flat_enc": flatness(cs.reshape(-1, ch)[:, 0], sr),
         "mse_dec": float(np.mean((x.astype(float) - y.astype(float)) ** 2))}
    if ch == 2:
        r["lr_corr_orig"] = float(np.corrcoef(x.reshape(-1, 2).T.astype(float))[0, 1])
        r["lr_corr_enc"] = float(np.corrcoef(cs.reshape(-1, 2).T.astype(float))[0, 1])
    # plaintext sensitivity (same key + nonce), 1 LSB change at random sample
    ps = {"xcpda": [], "chacha20": [], "aes_ctr": []}
    for _ in range(TRIALS):
        x2 = x.copy(); j = rng.integers(x.size); x2[j] = np.int16(x2[j] ^ 1)
        c2 = cipher_samples(encrypt_audio(KEY, x2, sr, ch, nonce=NONCE))
        ps["xcpda"].append(nscr_uaci(c, c2))
        for k in ("chacha20", "aes_ctr"):
            ps[k].append(nscr_uaci(stream_cipher(k, KEY, NONCE, x), stream_cipher(k, KEY, NONCE, x2)))
    r["pt_sens"] = {k: list(np.mean(v, axis=0)) for k, v in ps.items()}
    # key sensitivity: flip one random bit of the 256-bit key
    ks, wk = [], []
    for _ in range(TRIALS):
        kb = bytearray(KEY); bit = rng.integers(256); kb[bit // 8] ^= 1 << (bit % 8); k2 = bytes(kb)
        b2 = encrypt_audio(k2, x, sr, ch, nonce=NONCE); ks.append(nscr_uaci(c, cipher_samples(b2)))
        try:
            decrypt_audio(k2, blob); wk.append(None)
        except AuthError:
            yw = decrypt_audio(k2, blob, verify=False)
            wk.append((float(np.corrcoef(x.astype(float), yw.astype(float))[0, 1]), snr(x, yw)))
    r["key_sens"] = list(np.mean(ks, axis=0))
    r["wrong_key_rejected"] = sum(w is not None for w in wk)
    r["wrong_key_corr"] = float(np.mean([abs(w[0]) for w in wk])); r["wrong_key_snr"] = float(np.mean([w[1] for w in wk]))
    res["files"][name] = r
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, (dict, list))}, r["pt_sens"]["xcpda"], r["key_sens"])

# ---------------- ablation (stage contributions) ----------------
abl = {}
for name in ["Romans (voice)", "Synthetic stereo music", "Gong"]:
    sr, ch, x = corpus[name]; hdr = make_header(sr, ch, x.size, x.size)
    for label, st in [("Mask only (ChaCha20)", ("mask",)), ("Permutation + mask", ("perm", "mask")),
                      ("Diffusion + mask", ("diff", "mask")), ("Full XC-PDA", ("perm", "diff", "mask"))]:
        c1 = np.frombuffer(encrypt_chunk(KEY, NONCE, hdr, x, stages=st)[0], "<u2").astype(np.int64)
        vals = []
        for _ in range(10):
            x2 = x.copy(); j = rng.integers(x.size); x2[j] = np.int16(x2[j] ^ 1)
            c2 = np.frombuffer(encrypt_chunk(KEY, NONCE, hdr, x2, stages=st)[0], "<u2").astype(np.int64)
            vals.append(nscr_uaci(c1, c2))
        abl.setdefault(label, {})[name] = list(np.mean(vals, axis=0)) + [entropy(c1, 16), adj_corr(to_s(c1), ch)]
res["ablation"] = abl

# ---------------- nonce-misuse experiment ----------------
sr, ch, a = corpus["Romans (voice)"]; _, _, b = corpus["Train"]
n = min(a.size, b.size); a, b = a[:n], b[:n]
ca = cipher_samples(encrypt_audio(KEY, a, sr, 1, nonce=NONCE)); cb = cipher_samples(encrypt_audio(KEY, b, sr, 1, nonce=NONCE))
pa, pb = to_u16(a), to_u16(b)
sa, sb = stream_cipher("chacha20", KEY, NONCE, a), stream_cipher("chacha20", KEY, NONCE, b)
res["nonce_reuse"] = {
    "samples": int(n),
    "chacha20_match_pct": float(np.mean((sa ^ sb) == (pa ^ pb)) * 100),
    "xcpda_match_pct": float(np.mean((ca ^ cb) == (pa ^ pb)) * 100),
    "chacha20_corr": float(np.corrcoef((sa ^ sb).astype(float), (pa ^ pb).astype(float))[0, 1]),
    "xcpda_corr": float(np.corrcoef((ca ^ cb).astype(float), (pa ^ pb).astype(float))[0, 1]),
}
print(res["nonce_reuse"])

# ---------------- tamper detection ----------------
sr, ch, x = corpus["Synthetic speech"]
blob = encrypt_audio(KEY, x, sr, ch, chunk=sr, nonce=NONCE)
det = {"bit_flip": 0, "chunk_swap": 0, "truncation": 0, "header": 0}
T = 1000
for _ in range(T):
    k = rng.integers(len(blob["chunks"])); cbytes = bytearray(blob["chunks"][k]); p = rng.integers(len(cbytes) * 8)
    cbytes[p // 8] ^= 1 << (p % 8); bb = dict(blob, chunks=list(blob["chunks"])); bb["chunks"][k] = bytes(cbytes)
    try: decrypt_audio(KEY, bb)
    except AuthError: det["bit_flip"] += 1
bb = dict(blob, chunks=list(blob["chunks"]), tags=list(blob["tags"]))
bb["chunks"][0], bb["chunks"][1] = bb["chunks"][1], bb["chunks"][0]; bb["tags"][0], bb["tags"][1] = bb["tags"][1], bb["tags"][0]
try: decrypt_audio(KEY, bb)
except AuthError: det["chunk_swap"] = 1
bb = dict(blob, chunks=blob["chunks"][:-1], tags=blob["tags"][:-1])
try: decrypt_audio(KEY, bb)
except AuthError: det["truncation"] = 1
hh = bytearray(blob["header"]); hh[0] ^= 1; bb = dict(blob, header=bytes(hh))
try: decrypt_audio(KEY, bb)
except AuthError: det["header"] = 1
res["tamper"] = {"trials": T, **det, "chunks": len(blob["chunks"])}
print(res["tamper"])

# chunk-mode plaintext sensitivity (1 s frames)
c1 = cipher_samples(blob); v = []
for _ in range(10):
    x2 = x.copy(); j = rng.integers(x.size); x2[j] = np.int16(x2[j] ^ 1)
    c2 = cipher_samples(encrypt_audio(KEY, x2, sr, ch, chunk=sr, nonce=NONCE)); k = j // sr
    v.append(nscr_uaci(c1[k * sr:(k + 1) * sr], c2[k * sr:(k + 1) * sr]) + (float(np.mean(c1 != c2) * 100),))
res["chunk_sens"] = list(np.mean(v, axis=0))

# ---------------- throughput ----------------
long = synth_music_stereo(60).reshape(-1)          # 60 s, 44.1 kHz stereo = 10.58 MB
mb = long.size * 2 / 1e6
def bench(fn, reps=5):
    fn(); ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))
raw = long.astype("<i2").tobytes(); k32 = KEY; n12 = NONCE[:12]
thr = {}
thr["AES-256-CTR"] = bench(lambda: Cipher(algorithms.AES(k32), modes.CTR(NONCE[:16])).encryptor().update(raw))
thr["AES-256-GCM (AEAD)"] = bench(lambda: AESGCM(k32).encrypt(n12, raw, None))
thr["ChaCha20"] = bench(lambda: Cipher(algorithms.ChaCha20(k32, NONCE[:16]), None).encryptor().update(raw))
thr["ChaCha20-Poly1305 (AEAD)"] = bench(lambda: ChaCha20Poly1305(k32).encrypt(n12, raw, None))
timings = {}
thr["XC-PDA (whole file)"] = bench(lambda: encrypt_audio(KEY, long, 44100, 2, nonce=NONCE))
thr["XC-PDA (1 s frames)"] = bench(lambda: encrypt_audio(KEY, long, 44100, 2, chunk=88200, nonce=NONCE))
bl = encrypt_audio(KEY, long, 44100, 2, nonce=NONCE)
dec_t = bench(lambda: decrypt_audio(KEY, bl))
encrypt_audio(KEY, long, 44100, 2, nonce=NONCE, timings=timings)
res["throughput"] = {"MB": mb, "enc_seconds": thr, "MBps": {k: mb / v for k, v in thr.items()},
                     "xcpda_dec_seconds": dec_t, "xcpda_dec_MBps": mb / dec_t, "stage_seconds": timings,
                     "frame_latency_ms_1s": thr["XC-PDA (1 s frames)"] / 60 * 1000}
print(res["throughput"])

# ---------------- NIST SP 800-22 subset ----------------
long2 = synth_music_stereo(80).reshape(-1)
S, L, RUNS = 100, 1_000_000, 5
pv = {}; per_run = {}
for run in range(RUNS):
    kr = rng.bytes(32); nr = rng.bytes(24)
    cb = b"".join(encrypt_audio(kr, long2, 44100, 2, nonce=nr)["chunks"])
    bits = np.unpackbits(np.frombuffer(cb, np.uint8))
    for i in range(S):
        for k, p in nist.run_all(bits[i * L:(i + 1) * L]).items():
            pv.setdefault(k, []).append(float(p))
            per_run.setdefault(k, [0] * RUNS)[run] += int(p >= 0.01)
from scipy.special import gammaincc
nist_res = {}
for k, ps in pv.items():
    ps = np.array(ps); n_ = ps.size; h = np.histogram(ps, bins=10, range=(0, 1))[0]
    chi = ((h - n_ / 10) ** 2 / (n_ / 10)).sum()
    nist_res[k] = {"pass": int((ps >= 0.01).sum()), "uniformity_p": float(gammaincc(4.5, chi / 2)),
                   "mean_p": float(ps.mean()), "per_run": per_run[k]}
res["nist"] = {"sequences": S * RUNS, "runs": RUNS, "length": L, "tests": nist_res}
print(nist_res)

json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)

