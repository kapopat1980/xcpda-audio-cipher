# =====================================================================================
# XC-PDA  -  Kaggle experiment pack
#   Part A  Statistical / differential evaluation on standard audio datasets
#   Part B  Deep-learning leakage attack (GPU): can a CNN tell the source class from ciphertext?
#   Part C  Throughput on the Kaggle CPU
#   Part D  NIST SP 800-22 on real-dataset ciphertext (optional, CPU)
# Output: /kaggle/working/xcpda_results.zip  -> upload this file back to the chat.
#
# How to run on Kaggle
#   1. New Notebook -> Settings: Accelerator = GPU (T4 x2 or P100), Internet = On.
#   2. "+ Add Input" -> add audio datasets (search: "GTZAN music genre", "ESC-50",
#      "speech commands" or "LibriSpeech"). Any dataset with .wav/.flac files works.
#   3. Paste this whole file into one cell and run it. Expected run time 30-90 min.
# =====================================================================================
import os, sys, json, time, struct, hmac, glob, random, zipfile, platform, warnings
warnings.filterwarnings("ignore")
import subprocess
def _pip(p):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", p], check=False)
for mod, pkg in [("numba", "numba"), ("cryptography", "cryptography"), ("soundfile", "soundfile"), ("sklearn", "scikit-learn")]:
    try: __import__(mod)
    except ImportError: _pip(pkg)

import numpy as np
from numba import njit
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305, AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes, poly1305
from scipy.signal import welch
from scipy.special import erfc, gammaincc
from scipy.stats import norm, binomtest
import soundfile as sf

# ------------------------------- configuration -------------------------------------
INPUT_ROOT   = os.environ.get("XC_INPUT", "/kaggle/input")
OUT          = os.environ.get("XC_OUT", "/kaggle/working/xcpda_results")
FILES_PER_DS = int(os.environ.get("XC_FILES", 40))     # Part A: files sampled per dataset
TRIALS       = int(os.environ.get("XC_TRIALS", 5))     # Part A: sensitivity trials per file
DL_CLIP_SEC  = 1.0                                      # Part B: clip length (s) at 16 kHz
DL_MAX_PER_CLASS = int(os.environ.get("XC_PER_CLASS", 300))
DL_EPOCHS    = int(os.environ.get("XC_EPOCHS", 20))
RUN_NIST     = os.environ.get("XC_NIST", "1") == "1"
SEED = 2026
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED); random.seed(SEED)
LOG = open(f"{OUT}/log.txt", "a")
T_START = time.time()
def log(*a):
    s = f"[{(time.time() - T_START) / 60:6.1f} min] " + " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

# =============================== XC-PDA cipher ======================================
MAGIC = b"XC-PDA/v1"
def chacha_stream(key, nbytes, sid):
    nonce = struct.pack("<I", 0) + struct.pack("<Q", sid) + b"\x00" * 4
    return Cipher(algorithms.ChaCha20(key, nonce), mode=None).encryptor().update(b"\x00" * nbytes)
def derive(key, nonce, header, k, last):
    info = MAGIC + header + struct.pack("<QB", k, 1 if last else 0)
    o = HKDF(algorithm=hashes.SHA256(), length=128, salt=nonce, info=info).derive(key)
    return o[:32], o[32:64], o[64:96], o[96:]
@njit(cache=True)
def _perm(n, rnd):
    idx = np.arange(n)
    for i in range(n - 1, 0, -1):
        j = rnd[i] % np.uint64(i + 1); t = idx[i]; idx[i] = idx[j]; idx[j] = t
    return idx
@njit(cache=True)
def _rotl(x, r):
    r &= 15
    return ((x << r) | (x >> ((16 - r) & 15))) & 0xFFFF if r else x
@njit(cache=True)
def _rotr(x, r):
    r &= 15
    return ((x >> r) | (x << ((16 - r) & 15))) & 0xFFFF if r else x
@njit(cache=True)
def _dfwd(x, a, b, rr, ss, iv1, iv2):
    n = x.shape[0]; u = np.empty(n, np.int64); p = iv1
    for i in range(n):
        p = _rotl(((x[i] ^ a[i]) + p) & 0xFFFF, rr[i]); u[i] = p
    w = np.empty(n, np.int64); q = iv2
    for i in range(n - 1, -1, -1):
        q = _rotl(((u[i] ^ b[i]) + q) & 0xFFFF, ss[i]); w[i] = q
    return w
@njit(cache=True)
def _dinv(w, a, b, rr, ss, iv1, iv2):
    n = w.shape[0]; u = np.empty(n, np.int64)
    for i in range(n):
        nx = w[i + 1] if i + 1 < n else iv2
        u[i] = ((_rotr(w[i], ss[i]) - nx) & 0xFFFF) ^ b[i]
    x = np.empty(n, np.int64)
    for i in range(n):
        pv = u[i - 1] if i > 0 else iv1
        x[i] = ((_rotr(u[i], rr[i]) - pv) & 0xFFFF) ^ a[i]
    return x
def _dmat(kd, n):
    raw = np.frombuffer(chacha_stream(kd, 5 * n + 4, 2), np.uint8)
    a = raw[:2 * n].view("<u2").astype(np.int64); b = raw[2 * n:4 * n].view("<u2").astype(np.int64)
    rs = raw[4 * n:5 * n].astype(np.int64)
    return a, b, rs & 15, rs >> 4, int.from_bytes(bytes(raw[5 * n:5 * n + 2]), "little"), int.from_bytes(bytes(raw[5 * n + 2:5 * n + 4]), "little")
def _tag(km, h, nonce, k, last, c):
    p = poly1305.Poly1305(km); p.update(MAGIC + h + nonce + struct.pack("<QBQ", k, 1 if last else 0, len(c)) + c); return p.finalize()
def enc_chunk(key, nonce, h, s, k=0, last=True, stages=("perm", "diff", "mask")):
    kp, kd, kmk, kt = derive(key, nonce, h, k, last)
    x = s.astype(np.int64) & 0xFFFF; n = x.size
    if "perm" in stages: x = x[_perm(n, np.frombuffer(chacha_stream(kp, 8 * n, 1), "<u8"))]
    if "diff" in stages: x = _dfwd(x, *_dmat(kd, n))
    if "mask" in stages: x = x ^ np.frombuffer(chacha_stream(kmk, 2 * n, 3), "<u2").astype(np.int64)
    c = (x & 0xFFFF).astype("<u2").tobytes()
    return c, _tag(kt, h, nonce, k, last, c)
class AuthError(Exception): pass
def dec_chunk(key, nonce, h, c, t, k=0, last=True, verify=True):
    kp, kd, kmk, kt = derive(key, nonce, h, k, last)
    if verify and not hmac.compare_digest(_tag(kt, h, nonce, k, last, c), t): raise AuthError()
    y = np.frombuffer(c, "<u2").astype(np.int64); n = y.size
    y = _dinv(y ^ np.frombuffer(chacha_stream(kmk, 2 * n, 3), "<u2").astype(np.int64), *_dmat(kd, n))
    p = _perm(n, np.frombuffer(chacha_stream(kp, 8 * n, 1), "<u8")); x = np.empty(n, np.int64); x[p] = y
    return (x & 0xFFFF).astype(np.uint16).view(np.int16)
def header(sr, ch, n, L): return struct.pack("<IHHQI", sr, ch, 16, n, L)
def encrypt(key, s, sr, ch, chunk=None, nonce=None, stages=("perm", "diff", "mask")):
    n = s.size; chunk = chunk or n; nonce = nonce or os.urandom(24); h = header(sr, ch, n, chunk)
    m = (n + chunk - 1) // chunk; cs, ts = [], []
    for k in range(m):
        c, t = enc_chunk(key, nonce, h, s[k * chunk:(k + 1) * chunk], k, k == m - 1, stages); cs.append(c); ts.append(t)
    return {"h": h, "n": nonce, "c": cs, "t": ts}
def decrypt(key, b, verify=True):
    m = len(b["c"])
    return np.concatenate([dec_chunk(key, b["n"], b["h"], c, t, k, k == m - 1, verify) for k, (c, t) in enumerate(zip(b["c"], b["t"]))])
def csamples(b): return np.frombuffer(b"".join(b["c"]), "<u2").astype(np.int64)

# self-test
_k = os.urandom(32); _x = rng.integers(-32768, 32768, 10007).astype(np.int16)
assert np.array_equal(decrypt(_k, encrypt(_k, _x, 16000, 1, chunk=3000)), _x); log("XC-PDA self-test OK")

# =============================== audio loading =======================================
EXT = (".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif")
BAD_FILES = []
import signal
class FileTimeout(Exception): pass
def _alarm(signum, frame): raise FileTimeout("file read timed out")
def with_timeout(fn, *a, seconds=20):
    """Run fn(*a); give up after `seconds` so one bad file can never stall the run."""
    old_h = signal.signal(signal.SIGALRM, _alarm); signal.alarm(seconds)
    try: return fn(*a)
    finally: signal.alarm(0); signal.signal(signal.SIGALRM, old_h)
def to_int16(path, max_sec=30.0):
    """Return (int16 interleaved samples, sr, channels, source_bits). Keeps native sr and channels.
    Uses libsndfile only: an unreadable file raises at once instead of falling back to a decoder
    that can hang (e.g. the known-corrupt GTZAN jazz.00054.wav)."""
    info = sf.info(path)
    frames = int(min(info.frames, max_sec * info.samplerate))
    sub = info.subtype or ""
    if sub in ("PCM_16",):
        x = sf.read(path, frames=frames, dtype="int16", always_2d=True)[0]
    else:
        x = sf.read(path, frames=frames, dtype="float32", always_2d=True)[0]
        x = np.clip(np.round(x * 32767), -32768, 32767).astype(np.int16)
    bits = {"PCM_16": 16, "PCM_24": 24, "PCM_32": 32, "FLOAT": 32, "PCM_U8": 8, "PCM_S8": 8}.get(sub, 0)
    return x.reshape(-1), int(info.samplerate), x.shape[1], bits

def find_datasets(root):
    """Group audio files by dataset. Handles both Kaggle layouts:
       /kaggle/input/<slug>/...  and  /kaggle/input/datasets/<owner>/<slug>/..."""
    ds = {}
    for f in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if not f.lower().endswith(EXT): continue
        parts = os.path.relpath(f, root).split(os.sep)
        if parts[0] in ("datasets", "competitions", "models") and len(parts) > 3:
            key = parts[2]                        # the dataset slug
        else:
            key = parts[0]
        ds.setdefault(key, []).append(f)
    return {k: sorted(v) for k, v in sorted(ds.items())}
DATASETS = find_datasets(INPUT_ROOT)
for k, v in DATASETS.items(): log(f"dataset {k}: {len(v)} audio files")
if not DATASETS: raise SystemExit("No audio found under " + INPUT_ROOT + " - add a dataset with '+ Add Input'.")

# =============================== metrics =============================================
def entropy(v, bits):
    c = np.bincount(v, minlength=1 << bits).astype(float); p = c[c > 0] / c.sum(); return float(-(p * np.log2(p)).sum())
u16 = lambda x: x.astype(np.int64) & 0xFFFF
s16 = lambda x: ((x.astype(np.int64) + 32768) & 0xFFFF) - 32768
def adj_corr(x, ch):
    y = x.reshape(-1, ch)[:, 0].astype(float)
    if y.std() == 0: return float("nan")
    return float(np.corrcoef(y[:-1], y[1:])[0, 1])
def chi_bytes(u):
    c = np.bincount(np.frombuffer(u.astype("<u2").tobytes(), np.uint8), minlength=256); e = c.sum() / 256; return float(((c - e) ** 2 / e).sum())
def snr(o, d):
    o = o.astype(float); e = ((o - d.astype(float)) ** 2).sum(); return float(10 * np.log10((o ** 2).sum() / e)) if (o ** 2).sum() > 0 else float("nan")
def flat(x, sr):
    f, P = welch(x.astype(float), sr, nperseg=1024); P = P[1:] + 1e-30; return float(np.exp(np.mean(np.log(P))) / np.mean(P))
def nscr_uaci(a, b):
    d = np.abs(a.astype(np.int64) - b.astype(np.int64)); return float(np.mean(d != 0) * 100), float(np.mean(d) / 65535 * 100)
def stream(kind, key, nonce, x):
    data = u16(x).astype("<u2").tobytes()
    e = Cipher(algorithms.ChaCha20(key, nonce[:16]), None).encryptor() if kind == "chacha20" else Cipher(algorithms.AES(key), modes.CTR(nonce[:16])).encryptor()
    return np.frombuffer(e.update(data), "<u2").astype(np.int64)

# =============================== PART A ==============================================
import csv
KEY = bytes(range(32)); NONCE = bytes(range(100, 124))
rows = []
tA = time.time()
for dsname, files in DATASETS.items():
    pick = random.sample(files, min(FILES_PER_DS, len(files)))
    log(f"Part A: {dsname}: testing {len(pick)} files")
    for path_i, path in enumerate(pick):
        if path_i and path_i % 10 == 0: log(f"   {path_i}/{len(pick)} files done")
        try:
            x, sr, ch, bits = with_timeout(to_int16, path)
        except Exception as e:
            BAD_FILES.append(os.path.relpath(path, INPUT_ROOT)); log("   skipped unreadable file:", os.path.relpath(path, INPUT_ROOT), type(e).__name__); continue
        if x.size < 2048 or np.all(x == x[0]): continue
        b = encrypt(KEY, x, sr, ch, nonce=NONCE); c = csamples(b); y = decrypt(KEY, b); cs = s16(c)
        ideal = entropy(rng.integers(0, 65536, x.size), 16)
        r = dict(dataset=dsname, file=os.path.relpath(path, INPUT_ROOT), sr=sr, ch=ch, src_bits=bits, samples=int(x.size),
                 seconds=round(x.size / ch / sr, 3), bit_exact=bool(np.array_equal(x, y)),
                 H16_orig=entropy(u16(x), 16), H16_enc=entropy(c, 16), H16_ideal=ideal,
                 H8_orig=entropy(np.frombuffer(u16(x).astype("<u2").tobytes(), np.uint8), 8),
                 H8_enc=entropy(np.frombuffer(c.astype("<u2").tobytes(), np.uint8), 8),
                 r_orig=adj_corr(x, ch), r_enc=adj_corr(cs, ch), chi2_enc=chi_bytes(c), snr_enc=snr(x, cs),
                 sfm_orig=flat(x.reshape(-1, ch)[:, 0], sr), sfm_enc=flat(cs.reshape(-1, ch)[:, 0], sr))
        if ch == 2:
            r["lr_orig"] = float(np.corrcoef(x.reshape(-1, 2).T.astype(float))[0, 1]); r["lr_enc"] = float(np.corrcoef(cs.reshape(-1, 2).T.astype(float))[0, 1])
        ps, ks, cc = [], [], []
        for _ in range(TRIALS):
            x2 = x.copy(); j = rng.integers(x.size); x2[j] = np.int16(x2[j] ^ 1)
            ps.append(nscr_uaci(c, csamples(encrypt(KEY, x2, sr, ch, nonce=NONCE))))
            cc.append(nscr_uaci(stream("chacha20", KEY, NONCE, x), stream("chacha20", KEY, NONCE, x2))[0])
            kb = bytearray(KEY); bt = rng.integers(256); kb[bt // 8] ^= 1 << (bt % 8)
            ks.append(nscr_uaci(c, csamples(encrypt(bytes(kb), x, sr, ch, nonce=NONCE))))
        r.update(nscr=np.mean(ps, 0)[0], uaci=np.mean(ps, 0)[1], nscr_chacha=float(np.mean(cc)), key_nscr=np.mean(ks, 0)[0], key_uaci=np.mean(ks, 0)[1])
        try: decrypt(bytes(kb), b); r["wrong_key_rejected"] = False
        except AuthError: r["wrong_key_rejected"] = True
        rows.append(r)
    log(f"Part A: {dsname} done ({sum(1 for q in rows if q['dataset']==dsname)} files)")
keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0].keys()).index(k) if k in rows[0] else 99)
with open(f"{OUT}/partA_per_file.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
summ = {}
for dsname in DATASETS:
    R = [r for r in rows if r["dataset"] == dsname]
    if not R: continue
    agg = lambda k: [float(np.nanmin([r[k] for r in R])), float(np.nanmean([r[k] for r in R])), float(np.nanmax([r[k] for r in R]))]
    summ[dsname] = {"files": len(R), "bit_exact_all": all(r["bit_exact"] for r in R),
                    "sample_rates": sorted({r["sr"] for r in R}), "channels": sorted({r["ch"] for r in R}), "source_bits": sorted({r["src_bits"] for r in R}),
                    "seconds_total": float(sum(r["seconds"] for r in R)),
                    **{k + "_min_mean_max": agg(k) for k in ["H8_enc", "H16_enc", "r_orig", "r_enc", "chi2_enc", "snr_enc", "sfm_orig", "sfm_enc", "nscr", "uaci", "nscr_chacha", "key_nscr", "key_uaci"]},
                    "H16_gap_to_ideal_max": float(max(abs(r["H16_enc"] - r["H16_ideal"]) for r in R)),
                    "chi2_pass_rate": float(np.mean([r["chi2_enc"] < 293.25 for r in R])),
                    "wrong_key_rejected_all": all(r["wrong_key_rejected"] for r in R)}
json.dump(summ, open(f"{OUT}/partA_summary.json", "w"), indent=1)
log("Part A finished in", round(time.time() - tA), "s")

# =============================== PART B (GPU) =========================================
# Labelled clips: label = parent folder name (GTZAN, Speech Commands, ...) or ESC-50 metadata.
import torch, torch.nn as nn, torch.nn.functional as F
dev = "cuda" if torch.cuda.is_available() else "cpu"; log("Part B device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")
if dev == "cuda":
    _t = torch.randn(2048, 2048, device=dev); torch.cuda.synchronize(); log("GPU check OK:", float((_t @ _t).sum().item()) != 0.0)
elif os.environ.get("XC_ALLOW_CPU", "0") != "1":
    raise SystemExit("No GPU visible to PyTorch. In Kaggle: Settings -> Accelerator -> GPU T4 x2, then run again. (Set XC_ALLOW_CPU=1 to force a slow CPU run.)")
SR = 16000; L = int(SR * DL_CLIP_SEC)

def group_of(path, esc_fold):
    """Recordings that must never be split between train and test share a group id."""
    b = os.path.basename(path)
    if esc_fold is not None and b in esc_fold: return f"fold{esc_fold[b]}"   # ESC-50: official folds
    if "_nohash_" in b: return b.split("_nohash_")[0]                          # Speech Commands: speaker
    return os.path.splitext(b)[0]                                              # otherwise: the recording

def labelled_sets():
    """Returns {dataset: (pairs[(path, label, group)], info)} with duplicate copies removed."""
    out = {}
    for dsname, files in DATASETS.items():
        esc = [p for p in glob.glob(os.path.join(INPUT_ROOT, "**", "esc50.csv"), recursive=True) if f"{os.sep}{dsname}{os.sep}" in p]
        esc_fold = None
        if esc:
            import pandas as pd
            meta = pd.read_csv(esc[0]); lab = dict(zip(meta["filename"], meta["category"])); esc_fold = dict(zip(meta["filename"], meta["fold"]))
            pairs = [(f, lab[os.path.basename(f)]) for f in files if os.path.basename(f) in lab]
        else:
            pairs = [(f, os.path.basename(os.path.dirname(f))) for f in files]
            pairs = [(f, l) for f, l in pairs if not l.startswith("_")]
        # the same recording stored in several folders (e.g. ESC-50 at 16 kHz and 44.1 kHz) is kept once
        seen, uniq = set(), []
        for f, l in sorted(pairs, key=lambda q: (len(q[0]), q[0])):
            k = (l, os.path.basename(f))
            if k in seen: continue
            seen.add(k); uniq.append((f, l, group_of(f, esc_fold)))
        counts = {}
        for _, l, _ in uniq: counts[l] = counts.get(l, 0) + 1
        keep = {l for l, c in counts.items() if c >= 20}
        uniq = [q for q in uniq if q[1] in keep]
        if 2 <= len(keep) <= 60:
            out[dsname] = (uniq, {"files_found": len(pairs), "duplicates_removed": len(pairs) - len(seen),
                                  "split": "ESC-50 official fold 5 held out" if esc_fold else ("speaker-disjoint" if any("_nohash_" in q[0] for q in uniq) else "recording-disjoint")})
    return out

from math import gcd
from scipy.signal import resample_poly
def load_clip(path):
    info = sf.info(path)                                   # raises immediately on a corrupt file
    y = sf.read(path, frames=int(min(info.frames, (DL_CLIP_SEC + 2.0) * info.samplerate)), dtype="float32", always_2d=True)[0].mean(axis=1)
    if info.samplerate != SR:
        g = gcd(SR, int(info.samplerate)); y = resample_poly(y, SR // g, int(info.samplerate) // g).astype(np.float32)
    if y.size < L: y = np.pad(y, (0, L - y.size))
    s = int((y.size - L) / 2); y = y[s:s + L]
    return np.clip(np.round(y * 32767), -32768, 32767).astype(np.int16)

CONDITIONS = ["plaintext", "permutation_only_fixed_key", "chacha20_nonce_reused", "xcpda_nonce_reused", "xcpda_random_nonce"]
def transform(cond, x, idx):
    if cond == "plaintext": return x
    if cond == "permutation_only_fixed_key":
        return enc_view(encrypt(KEY, x, SR, 1, nonce=NONCE, stages=("perm",)))
    if cond == "chacha20_nonce_reused": return s16(stream("chacha20", KEY, NONCE, x)).astype(np.int16)
    if cond == "xcpda_nonce_reused": return enc_view(encrypt(KEY, x, SR, 1, nonce=NONCE))
    if cond == "xcpda_random_nonce": return enc_view(encrypt(KEY, x, SR, 1))
def enc_view(b): return s16(csamples(b)).astype(np.int16)

class M5(nn.Module):   # raw-waveform CNN (Dai et al. style), strong on plaintext audio
    def __init__(s, n_cls, ch=64):
        super().__init__()
        s.c1 = nn.Conv1d(1, ch, 80, 4); s.b1 = nn.BatchNorm1d(ch)
        s.c2 = nn.Conv1d(ch, ch, 3); s.b2 = nn.BatchNorm1d(ch)
        s.c3 = nn.Conv1d(ch, 2 * ch, 3); s.b3 = nn.BatchNorm1d(2 * ch)
        s.c4 = nn.Conv1d(2 * ch, 2 * ch, 3); s.b4 = nn.BatchNorm1d(2 * ch)
        s.fc = nn.Linear(2 * ch, n_cls)
    def forward(s, x):
        for c, b in [(s.c1, s.b1), (s.c2, s.b2), (s.c3, s.b3), (s.c4, s.b4)]:
            x = F.max_pool1d(F.relu(b(c(x))), 4)
        return s.fc(F.adaptive_avg_pool1d(x, 1).squeeze(-1))

class SpecCNN(nn.Module):  # log-spectrogram 2D CNN
    def __init__(s, n_cls):
        super().__init__()
        s.f = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        s.fc = nn.Linear(128, n_cls)
    def forward(s, x):
        S = torch.stft(x.squeeze(1), 512, 256, window=torch.hann_window(512, device=x.device), return_complex=True).abs()
        return s.fc(s.f(torch.log1p(S).unsqueeze(1)).flatten(1))

def train_eval(model_fn, Xtr, ytr, Xte, yte, n_cls):
    torch.manual_seed(SEED); m = model_fn(n_cls).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=DL_EPOCHS * ((len(Xtr) + 63) // 64))
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32).unsqueeze(1) / 32768; Xte_t = torch.tensor(Xte, dtype=torch.float32).unsqueeze(1) / 32768
    ytr_t = torch.tensor(ytr); best = 0
    for ep in range(DL_EPOCHS):
        m.train(); perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), 64):
            bi = perm[i:i + 64]; xb = Xtr_t[bi].to(dev); yb = ytr_t[bi].to(dev)
            loss = F.cross_entropy(m(xb), yb); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    m.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(Xte_t), 256): preds.append(m(Xte_t[i:i + 256].to(dev)).argmax(1).cpu())
    p = torch.cat(preds).numpy()
    from sklearn.metrics import f1_score
    acc = float((p == yte).mean()); f1 = float(f1_score(yte, p, average="macro"))
    return acc, f1

def hist_attack(Xtr, ytr, Xte, yte):   # classical attacker: 256-bin byte histogram + logistic regression
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    feat = lambda X: np.stack([np.bincount(np.frombuffer(u16(x).astype("<u2").tobytes(), np.uint8), minlength=256) / (2 * x.size) for x in X])
    sc = StandardScaler().fit(feat(Xtr)); clf = LogisticRegression(max_iter=2000).fit(sc.transform(feat(Xtr)), ytr)
    return float((clf.predict(sc.transform(feat(Xte))) == yte).mean())

dl = {}
tB = time.time()
import hashlib
from sklearn.model_selection import GroupShuffleSplit
for dsname, (pairs, info) in labelled_sets().items():
    labels = sorted({l for _, l, _ in pairs}); lid = {l: i for i, l in enumerate(labels)}
    by_cls = {}
    for q in pairs: by_cls.setdefault(q[1], []).append(q)
    clips, ys, groups, seen_h = [], [], [], set(); content_dups = 0
    for l in labels:
        for f, _, g in random.sample(by_cls[l], min(DL_MAX_PER_CLASS, len(by_cls[l]))):
            try: c = with_timeout(load_clip, f)
            except Exception as e:
                BAD_FILES.append(os.path.relpath(f, INPUT_ROOT)); log("   skipped unreadable file:", os.path.relpath(f, INPUT_ROOT), type(e).__name__); continue
            if (len(clips) + 1) % 2000 == 0: log(f"   loaded {len(clips) + 1} clips")
            h = hashlib.sha256(c.tobytes()).hexdigest()
            if h in seen_h: content_dups += 1; continue          # identical audio kept once
            seen_h.add(h); clips.append(c); ys.append(lid[l]); groups.append(g)
    clips = np.stack(clips); ys = np.array(ys); groups = np.array(groups)
    if info["split"].startswith("ESC-50"):
        te_mask = groups == "fold5"; tr, te = np.where(~te_mask)[0], np.where(te_mask)[0]
    else:
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(clips, ys, groups))
    assert not (set(groups[tr]) & set(groups[te])), "group leak between train and test"
    chance = float(np.bincount(ys[te], minlength=len(labels)).max() / len(te))
    log(f"Part B: {dsname}: {len(labels)} classes, {len(ys)} unique clips ({info['duplicates_removed']} duplicate files and {content_dups} identical clips removed), "
        f"split={info['split']}, train={len(tr)} test={len(te)}, majority-class chance {chance:.3f}")
    res = {"classes": len(labels), "clips": int(len(ys)), "train_clips": int(len(tr)), "test_clips": int(len(te)), "chance": chance,
           "split": info["split"], "duplicate_files_removed": info["duplicates_removed"], "identical_clips_removed": content_dups,
           "groups_train": int(len(set(groups[tr]))), "groups_test": int(len(set(groups[te]))), "conditions": {}}
    for cond in CONDITIONS:
        X = np.stack([transform(cond, x, i) for i, x in enumerate(clips)])
        r = {}
        for name, fn in [("raw_waveform_cnn", M5), ("spectrogram_cnn", SpecCNN)]:
            t0 = time.time(); acc, f1 = train_eval(fn, X[tr], ys[tr], X[te], ys[te], len(labels))
            log(f"      {cond} / {name}: acc {acc:.3f} ({time.time() - t0:.0f} s)")
            k = int(round(acc * len(te)))
            r[name] = {"acc": acc, "macro_f1": f1, "p_vs_chance": float(binomtest(k, len(te), chance, alternative="greater").pvalue)}
        acc = hist_attack(X[tr], ys[tr], X[te], ys[te]); k = int(round(acc * len(te)))
        r["byte_histogram_logreg"] = {"acc": acc, "p_vs_chance": float(binomtest(k, len(te), chance, alternative="greater").pvalue)}
        res["conditions"][cond] = r
        log(f"   {cond:28s} " + "  ".join(f"{n}={v['acc']:.3f}" for n, v in r.items()))
    dl[dsname] = res
    json.dump(dl, open(f"{OUT}/partB_dl_attack.json", "w"), indent=1)
log("Part B finished in", round(time.time() - tB), "s")

# =============================== PART C ==============================================
def _safe(f):
    try: return with_timeout(to_int16, f)[0]
    except Exception: return np.zeros(0, np.int16)
long = np.concatenate([_safe(f) for f in random.sample(sum(DATASETS.values(), []), min(200, sum(len(v) for v in DATASETS.values())))])[:44100 * 2 * 60]
if long.size < 44100 * 2 * 60: long = np.resize(long, 44100 * 2 * 60)
raw = long.astype("<i2").tobytes(); mb = len(raw) / 1e6
def bench(fn, reps=5):
    fn(); ts = []
    for _ in range(reps): t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))
thr = {"AES-256-CTR": bench(lambda: Cipher(algorithms.AES(KEY), modes.CTR(NONCE[:16])).encryptor().update(raw)),
       "AES-256-GCM": bench(lambda: AESGCM(KEY).encrypt(NONCE[:12], raw, None)),
       "ChaCha20": bench(lambda: Cipher(algorithms.ChaCha20(KEY, NONCE[:16]), None).encryptor().update(raw)),
       "ChaCha20-Poly1305": bench(lambda: ChaCha20Poly1305(KEY).encrypt(NONCE[:12], raw, None)),
       "XC-PDA whole file": bench(lambda: encrypt(KEY, long, 44100, 2, nonce=NONCE)),
       "XC-PDA 1 s frames": bench(lambda: encrypt(KEY, long, 44100, 2, chunk=88200, nonce=NONCE))}
blob = encrypt(KEY, long, 44100, 2, nonce=NONCE); thr["XC-PDA decrypt whole file"] = bench(lambda: decrypt(KEY, blob))
cpu = ""
try: cpu = [l.split(":")[1].strip() for l in open("/proc/cpuinfo") if "model name" in l][0]
except Exception: pass
json.dump({"cpu": cpu, "MB": mb, "seconds": thr, "MBps": {k: mb / v for k, v in thr.items()}}, open(f"{OUT}/partC_throughput.json", "w"), indent=1)
log("Part C:", {k: round(mb / v, 1) for k, v in thr.items()}, cpu)

# =============================== PART D (optional) ===================================
def nist_suite(b):
    n = b.size; X = 2 * b.astype(np.int64) - 1; out = {}
    out["monobit"] = erfc(abs(X.sum()) / np.sqrt(n) / np.sqrt(2))
    M = 128; N = n // M; pi = b[:N * M].reshape(N, M).mean(1); out["block_freq"] = gammaincc(N / 2, 4 * M * np.sum((pi - .5) ** 2) / 2)
    p = b.mean(); V = 1 + np.count_nonzero(b[1:] != b[:-1])
    out["runs"] = 0.0 if abs(p - .5) >= 2 / np.sqrt(n) else erfc(abs(V - 2 * n * p * (1 - p)) / (2 * np.sqrt(2 * n) * p * (1 - p)))
    pis = np.array([.0882, .2092, .2483, .1933, .1208, .0675, .0727]); v = np.zeros(7)
    for blk in b[:75 * 10000].reshape(75, 10000):
        d = np.diff(np.concatenate(([0], blk.astype(np.int8), [0]))); s_, e_ = np.where(d == 1)[0], np.where(d == -1)[0]
        Lr = (e_ - s_).max() if s_.size else 0; v[min(max(Lr, 10), 16) - 10] += 1
    out["longest_run"] = gammaincc(3, np.sum((v - 75 * pis) ** 2 / (75 * pis)) / 2)
    m_ = np.abs(np.fft.rfft(X.astype(float)))[:n // 2]; T = np.sqrt(np.log(20) * n)
    out["dft"] = erfc(abs((np.count_nonzero(m_ < T) - .95 * n / 2) / np.sqrt(n * .95 * .05 / 4)) / np.sqrt(2))
    def cnt(mm):
        ext = np.concatenate((b, b[:mm - 1])).astype(np.int64); val = np.zeros(n, np.int64)
        for k in range(mm): val = (val << 1) | ext[k:k + n]
        return np.bincount(val, minlength=1 << mm)
    def phi(mm): c = cnt(mm) / n; c = c[c > 0]; return np.sum(c * np.log(c))
    out["approx_entropy"] = gammaincc(2 ** 9, 2 * n * (np.log(2) - (phi(10) - phi(11))) / 2)
    psi = lambda mm: (2 ** mm / n) * np.sum(cnt(mm).astype(float) ** 2) - n
    a0, a1, a2 = psi(16), psi(15), psi(14)
    out["serial1"] = gammaincc(2 ** 14, (a0 - a1) / 2); out["serial2"] = gammaincc(2 ** 13, (a0 - 2 * a1 + a2) / 2)
    S = np.cumsum(X); z = np.max(np.abs(S)); sq = np.sqrt(n)
    k1 = np.arange(int((-n / z + 1) / 4), int((n / z - 1) / 4) + 1); k2 = np.arange(int((-n / z - 3) / 4), int((n / z - 1) / 4) + 1)
    out["cusum"] = float(1 - np.sum(norm.cdf((4 * k1 + 1) * z / sq) - norm.cdf((4 * k1 - 1) * z / sq)) + np.sum(norm.cdf((4 * k2 + 3) * z / sq) - norm.cdf((4 * k2 + 1) * z / sq)))
    return out
if RUN_NIST:
    tD = time.time(); pv = {}; S_ = 0
    allfiles = sum(DATASETS.values(), []); random.shuffle(allfiles); buf = b""
    for f in allfiles:
        if S_ >= 1000: break
        x = _safe(f)
        if x.size < 1000: continue
        buf += b"".join(encrypt(os.urandom(32), x, 16000, 1)["c"])
        while len(buf) >= 125000 and S_ < 1000:
            bits = np.unpackbits(np.frombuffer(buf[:125000], np.uint8)); buf = buf[125000:]
            for k, p in nist_suite(bits).items(): pv.setdefault(k, []).append(float(p))
            S_ += 1
    nres = {}
    for k, ps in pv.items():
        ps = np.array(ps); h = np.histogram(ps, 10, (0, 1))[0]; chi = ((h - ps.size / 10) ** 2 / (ps.size / 10)).sum()
        nres[k] = {"sequences": int(ps.size), "pass": int((ps >= 0.01).sum()),
                   "min_pass": int(np.ceil(ps.size * (0.99 - 3 * np.sqrt(0.0099 / ps.size)))), "uniformity_p": float(gammaincc(4.5, chi / 2))}
    json.dump(nres, open(f"{OUT}/partD_nist.json", "w"), indent=1)
    log("Part D:", {k: (v["pass"], v["sequences"]) for k, v in nres.items()}, round(time.time() - tD), "s")

# =============================== package =============================================
json.dump({"unreadable_files": sorted(set(BAD_FILES)), "python": sys.version, "platform": platform.platform(), "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None,
           "config": dict(FILES_PER_DS=FILES_PER_DS, TRIALS=TRIALS, DL_EPOCHS=DL_EPOCHS, DL_MAX_PER_CLASS=DL_MAX_PER_CLASS)},
          open(f"{OUT}/environment.json", "w"), indent=1)
LOG.close()
zp = OUT.rstrip("/") + ".zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(f"{OUT}/*"): z.write(f, os.path.basename(f))
print("\nDONE ->", zp, "  (download it from the Output panel and upload it to the chat)")
