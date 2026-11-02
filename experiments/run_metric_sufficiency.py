"""Do the usual audio-encryption metrics certify security?

Runs the statistical and differential metrics reported by recent audio-cipher papers (entropy, adjacent
correlation, histogram chi-square, NSCR/UACI for a one-LSB plaintext change, key sensitivity and nine NIST
SP 800-22 statistics) on five ciphers over the ten-file test corpus:
  XC-PDA (this work), AES-256-GCM-SIV (RFC 8452), Adiantum (XChaCha12 + AES-256, whole file as one message),
  ChaCha20 (secure stream cipher), and WK-20: XC-PDA keyed from a 20-bit secret (deliberately broken).
WK-20 is then broken by exhaustive search over its 2^20 keys using only the Poly1305 tag (ciphertext only).

Usage (repository root): python experiments/run_metric_sufficiency.py [--gallery DIR] [--out results/local]"""
import argparse, os, sys, json, time, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--gallery", default="/usr/lib/libreoffice/share/gallery/sounds")
ap.add_argument("--out", default="results/local")
ap.add_argument("--trials", type=int, default=20)
args = ap.parse_args()
import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter
from scipy.special import gammaincc
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.primitives.ciphers.aead import AESGCMSIV
from xcpda import encrypt_audio, cipher_samples
from xcpda.cipher import derive_subkeys, _tag
import adiantum, nist

rng = np.random.default_rng(2026)


# ---------------------------------------------------------------- corpus (identical to run_local_experiments.py)
def synth_music_stereo(sec, sr=44100):
    t = np.arange(int(sec * sr)) / sr
    chords = [[261.63, 329.63, 392.0], [220.0, 261.63, 329.63], [174.61, 220.0, 261.63], [196.0, 246.94, 293.66]]
    L = np.zeros_like(t); R = np.zeros_like(t)
    for k in range(int(sec / 0.5)):
        s = int(k * 0.5 * sr); e = min(int((k + 1) * 0.5 * sr) + sr // 4, t.size); tt = t[s:e] - t[s]
        env = np.exp(-3.0 * tt)
        for j, f in enumerate(chords[(k // 4) % 4]):
            note = sum(np.sin(2 * np.pi * f * h * tt) / h ** 1.3 for h in range(1, 7)) * env
            pan = 0.3 + 0.2 * j; L[s:e] += (1 - pan) * note; R[s:e] += pan * note
    L += 0.02 * rng.standard_normal(t.size); R += 0.02 * rng.standard_normal(t.size)
    m = max(np.abs(L).max(), np.abs(R).max())
    return (np.stack([L, R], axis=1) / m * 0.85 * 32767).astype(np.int16)


def synth_speech(sec, sr=44100):
    t = np.arange(int(sec * sr)) / sr
    f0 = 120 + 25 * np.sin(2 * np.pi * 0.7 * t) + 10 * np.sin(2 * np.pi * 3.1 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    src = sum(np.sin(h * phase) / h for h in range(1, 40))
    vowels = [(730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240), (530, 1840, 2480), (660, 1720, 2410)]
    out = np.zeros_like(t); seg = int(0.18 * sr)
    for k in range(t.size // seg):
        s, e = k * seg, (k + 1) * seg
        if k % 6 == 5: continue
        y = src[s:e]
        for F, bw in zip(vowels[k % 5], (80, 100, 120)):
            r = np.exp(-np.pi * bw / sr); th = 2 * np.pi * F / sr
            y = lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], y)
        out[s:e] = y * np.hanning(e - s)
    out += 0.003 * rng.standard_normal(t.size) * out.std()
    return (out / np.abs(out).max() * 0.8 * 32767).astype(np.int16)


GAL = args.gallery.rstrip("/") + "/"
corpus = {}
for name, f in [("Applause", "applause"), ("Drama (music)", "drama"), ("Gong", "gong"), ("Nature", "nature1"),
                ("Kongas (percussion)", "kongas"), ("Romans (voice)", "romans"), ("Train", "train"), ("Untie (effect)", "untie")]:
    sr, x = wavfile.read(GAL + f + ".wav"); corpus[name] = (sr, 1, x.astype(np.int16))
corpus["Synthetic speech"] = (44100, 1, synth_speech(5))
corpus["Synthetic stereo music"] = (44100, 2, synth_music_stereo(10).reshape(-1))


# ---------------------------------------------------------------- ciphers: (key, nonce, int16 samples) -> uint16 ciphertext samples
def u16(b): return np.frombuffer(b, "<u2").astype(np.int64)
def pbytes(x): return x.astype("<i2").tobytes()

def c_xcpda(k, n, x, sr, ch): return cipher_samples(encrypt_audio(k, x, sr, ch, nonce=n))
def c_siv(k, n, x, sr, ch): return u16(AESGCMSIV(k).encrypt(n[:12], pbytes(x), None)[:-16])
def c_adiantum(k, n, x, sr, ch): return u16(adiantum.encrypt(k, n + bytes(8), pbytes(x)))
def c_chacha(k, n, x, sr, ch): return u16(Cipher(algorithms.ChaCha20(k, n[:16]), None).encryptor().update(pbytes(x)))
def wk_key(seed): return hashlib.sha256(b"WK-20" + seed.to_bytes(4, "little")).digest()

CIPHERS = {"XC-PDA (this work)": c_xcpda, "AES-256-GCM-SIV": c_siv, "Adiantum (XChaCha12-AES)": c_adiantum,
           "ChaCha20 (stream)": c_chacha, "WK-20 (XC-PDA, 20-bit key)": c_xcpda}


# ---------------------------------------------------------------- metrics
def entropy(v, bits):
    c = np.bincount(v, minlength=1 << bits).astype(float); p = c[c > 0] / c.sum(); return float(-(p * np.log2(p)).sum())
def adj_corr(c, ch):
    s = (((c + 32768) & 0xFFFF) - 32768).reshape(-1, ch)[:, 0].astype(float); return float(np.corrcoef(s[:-1], s[1:])[0, 1])
def chi_bytes(c):
    cnt = np.bincount(np.frombuffer(c.astype("<u2").tobytes(), np.uint8), minlength=256); e = cnt.sum() / 256
    return float(((cnt - e) ** 2 / e).sum())
def nscr_uaci(a, b):
    d = np.abs(a - b); return float(np.mean(d != 0) * 100), float(np.mean(d) / 65535 * 100)


def keys_for(cname):
    """fresh key per use; WK-20 keys come from a random 20-bit seed"""
    if cname.startswith("WK-20"):
        return wk_key(int(rng.integers(1 << 20)))
    return rng.bytes(32)


res = {"trials": args.trials, "ciphers": {}}
for cname, enc in CIPHERS.items():
    per = {}
    t0 = time.time()
    for fname, (sr, ch, x) in corpus.items():
        k = keys_for(cname); n = rng.bytes(24)
        c = enc(k, n, x, sr, ch)
        H8 = entropy(np.frombuffer(c.astype("<u2").tobytes(), np.uint8), 8)
        ps = []
        for _ in range(args.trials):
            x2 = x.copy(); j = rng.integers(x.size); x2[j] = np.int16(x2[j] ^ 1)
            ps.append(nscr_uaci(c, enc(k, n, x2, sr, ch)))
        kb = bytearray(k); bit = int(rng.integers(256)); kb[bit // 8] ^= 1 << (bit % 8)
        per[fname] = {"H8": H8, "H16": entropy(c, 16), "corr": adj_corr(c, ch), "chi2": chi_bytes(c),
                      "nscr": float(np.mean([p[0] for p in ps])), "nscr_min": float(np.min([p[0] for p in ps])),
                      "uaci": float(np.mean([p[1] for p in ps])),
                      "key_nscr": nscr_uaci(c, enc(bytes(kb), n, x, sr, ch))[0]}
    # NIST: 100 sequences of 10^6 bits from 80 s of synthetic stereo music, five keys
    long2 = synth_music_stereo(80).reshape(-1)
    pv = {}
    for run in range(5):
        k = keys_for(cname); n = rng.bytes(24)
        bits = np.unpackbits(np.frombuffer(enc(k, n, long2, 44100, 2).astype("<u2").tobytes(), np.uint8))
        for i in range(20):
            for t, p in nist.run_all(bits[i * 1_000_000:(i + 1) * 1_000_000]).items():
                pv.setdefault(t, []).append(float(p))
    nres = {}
    for t, ps in pv.items():
        ps = np.array(ps); h = np.histogram(ps, bins=10, range=(0, 1))[0]
        chi = ((h - ps.size / 10) ** 2 / (ps.size / 10)).sum()
        nres[t] = {"pass": int((ps >= 0.01).sum()), "n": int(ps.size), "uniformity_p": float(gammaincc(4.5, chi / 2))}
    v = list(per.values())
    summary = {"H8_min": min(r["H8"] for r in v), "H8_max": max(r["H8"] for r in v),
               "corr_absmax": max(abs(r["corr"]) for r in v),
               "chi2_below_293": sum(r["chi2"] < 293.25 for r in v), "files": len(v),
               "nscr_min": min(r["nscr_min"] for r in v), "nscr_mean": float(np.mean([r["nscr"] for r in v])),
               "uaci_mean": float(np.mean([r["uaci"] for r in v])),
               "key_nscr_mean": float(np.mean([r["key_nscr"] for r in v])),
               "nist_passed": sum(r["pass"] >= 96 and r["uniformity_p"] >= 1e-4 for r in nres.values()), "nist_tests": len(nres),
               "nist_min_pass": min(r["pass"] for r in nres.values())}
    res["ciphers"][cname] = {"summary": summary, "files": per, "nist": nres, "seconds": time.time() - t0}
    print(cname, {k: (round(val, 4) if isinstance(val, float) else val) for k, val in summary.items()}, flush=True)

# ---------------------------------------------------------------- break WK-20 by exhaustive key search (ciphertext only)
sr, ch, x = corpus["Romans (voice)"]
seed = int(rng.integers(1 << 20)); n = rng.bytes(24)
blob = encrypt_audio(wk_key(seed), x, sr, ch, nonce=n)
hdr, chunk, tag = blob["header"], blob["chunks"][0], blob["tags"][0]
t0 = time.time(); found = None; tried = 0
for s in range(1 << 20):
    tried += 1
    kmac = derive_subkeys(wk_key(s), n, hdr, 0, True)[3]
    if _tag(kmac, hdr, n, 0, True, chunk) == tag:
        found = s; break
el = time.time() - t0
res["wk20_break"] = {"seed": seed, "found": found, "keys_tried": tried, "seconds": el,
                     "full_search_seconds_est": el / tried * (1 << 20), "ciphertext_bytes": len(chunk)}
print("WK-20 break", res["wk20_break"], flush=True)
# ---------------------------------------------------------------- Adiantum under a repeated tweak (romans.wav and train.wav, first 28,336 samples)
_, _, a = corpus["Romans (voice)"]; _, _, b = corpus["Train"]; n_ = 28336
pa, pb = a[:n_].astype("<i2"), b[:n_].astype("<i2"); k = bytes(range(32)); tw = bytes(32)
ca = np.frombuffer(adiantum.encrypt(k, tw, pa.tobytes()), "<u2"); cb = np.frombuffer(adiantum.encrypt(k, tw, pb.tobytes()), "<u2")
res["adiantum_tweak_reuse_xor_match_pct"] = float(np.mean((ca ^ cb) == (pa.view("<u2") ^ pb.view("<u2"))) * 100)
print("Adiantum tweak reuse", res["adiantum_tweak_reuse_xor_match_pct"])
os.makedirs(args.out, exist_ok=True)
json.dump(res, open(os.path.join(args.out, "metric_sufficiency.json"), "w"), indent=1)
