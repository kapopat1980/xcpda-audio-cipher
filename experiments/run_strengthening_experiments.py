"""Additional experiments for the revised manuscript.
  E1  AES-GCM-SIV (RFC 8452) as a misuse-resistant baseline: plaintext sensitivity, nonce reuse
  E2  Related-plaintext analysis of XC-PDA under nonce reuse (LSB carry relation of the add-rotate chain)
  E3  Frame length: latency, throughput, and recovery under packet loss / bit errors
  E4  Encryption of digital silence (all-zero input)
Throughput with AES instructions disabled is measured separately by bench_aead_baselines.py.
Usage (repository root): python experiments/run_strengthening_experiments.py [--out results/local]"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCMSIV, ChaCha20Poly1305
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from xcpda import encrypt_audio, decrypt_audio, decrypt_chunk, cipher_samples, encrypt_chunk, make_header, AuthError

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/local"); args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
rng = np.random.default_rng(2027)
KEY = bytes(range(32)); NONCE = bytes(range(100, 124))
res = {}

def synth_music(sec, sr=44100, seed=0):
    r = np.random.default_rng(seed); t = np.arange(int(sec * sr)) / sr
    chords = [[261.63, 329.63, 392.0], [220.0, 261.63, 329.63], [174.61, 220.0, 261.63], [196.0, 246.94, 293.66]]
    y = np.zeros_like(t)
    for k in range(int(sec / 0.5)):
        s = int(k * 0.5 * sr); e = min(s + int(0.75 * sr), t.size); tt = t[s:e] - t[s]
        for f in chords[(k // 4) % 4]:
            y[s:e] += sum(np.sin(2 * np.pi * f * h * tt) / h ** 1.3 for h in range(1, 7)) * np.exp(-3 * tt)
    y += 0.02 * r.standard_normal(t.size)
    return (y / np.abs(y).max() * 0.85 * 32767).astype(np.int16)

def nscr_uaci(a, b):
    d = np.abs(a.astype(np.int64) - b.astype(np.int64)); return float(np.mean(d != 0) * 100), float(np.mean(d) / 65535 * 100)

def u16(b): return np.frombuffer(b, "<u2").astype(np.int64)

# ------------------------------------------------------------------ E1 AES-GCM-SIV baseline
x = synth_music(10)                                  # 441,000 samples, mono 44.1 kHz
siv = AESGCMSIV(KEY); n12 = NONCE[:12]
def siv_ct(sig): return u16(siv.encrypt(n12, sig.astype("<i2").tobytes(), None)[:-16])
c0 = siv_ct(x); sens = []
for _ in range(20):
    x2 = x.copy(); j = rng.integers(x.size); x2[j] = np.int16(x2[j] ^ 1); sens.append(nscr_uaci(c0, siv_ct(x2)))
y = synth_music(10, seed=1)
p0, p1 = x.astype(np.int64) & 0xFFFF, y.astype(np.int64) & 0xFFFF
c1 = siv_ct(y)
res["E1_aes_gcm_siv"] = {"samples": int(x.size), "nscr_uaci_mean": list(np.mean(sens, 0)),
                          "nonce_reuse_xor_match_pct": float(np.mean((c0 ^ c1) == (p0 ^ p1)) * 100),
                          "identical_plaintext_identical_ciphertext": bool(np.array_equal(c0, siv_ct(x)))}
print("E1", res["E1_aes_gcm_siv"])

# ------------------------------------------------------------------ E2 related-plaintext analysis under nonce reuse
# Bit 0 of a modular sum is the XOR of the operands' bits 0, and the rotation moves it to bit s_i. For two frames that
# share the forward-chain value u_i (i before the first permuted difference), the ciphertext differences therefore satisfy
#   dW_i[s_i] = dW_{i+1}[0]     for every such i, with the same secret s_i in every pair (key and nonce reused).
N = 4096; hdr = make_header(16000, 1, N, N); base = rng.integers(-32768, 32768, N).astype(np.int16)
cb = u16(encrypt_chunk(KEY, NONCE, hdr, base)[0])
pairs = []
for m in range(64):
    x2 = base.copy(); x2[N - 1 - (m % 8)] ^= np.int16(1 << int(rng.integers(0, 15)))   # change near the end of the input
    pairs.append(cb ^ u16(encrypt_chunk(KEY, NONCE, hdr, x2)[0]))
D = np.stack(pairs)                                                # 64 x N ciphertext differences
bit = lambda v, s: (v >> s) & 1
# attacker: for each position i, find a rotation s consistent with all pairs
consistent = np.zeros(N - 1, bool); recovered = np.full(N - 1, -1)
for i in range(N - 1):
    for s in range(16):
        if np.all(bit(D[:, i], s) == bit(D[:, i + 1], 0)):
            consistent[i] = True; recovered[i] = s; break
# ground truth rotation amounts s_i of the diffusion layer for this key, nonce and header
from xcpda.cipher import derive_subkeys, _diff_material
_, kd, _, _ = derive_subkeys(KEY, NONCE, hdr, 0, True); a_, b_, rr, ss, iv1, iv2 = _diff_material(kd, N)
# positions where every pair still shares u_i: before the earliest permuted difference position across pairs
from xcpda.cipher import chacha_stream, _perm_indices
kp = derive_subkeys(KEY, NONCE, hdr, 0, True)[0]
perm = _perm_indices(N, np.frombuffer(chacha_stream(kp, 8 * N, 1), "<u8")); inv = np.argsort(perm)
first_diff = min(inv[N - 1 - k] for k in range(8))
pre = np.arange(first_diff)
# control: the same test on unrelated random data
R = rng.integers(0, 65536, (64, N))
ctrl = np.mean([any(np.all(bit(R[:, i], s) == bit(R[:, i + 1], 0)) for s in range(16)) for i in range(2000)])
res["E2_related_plaintext"] = {
    "pairs": 64, "frame_samples": N, "positions_before_first_difference": int(first_diff),
    "relation_holds_pct_before_first_difference": float(np.mean(consistent[pre]) * 100),
    "rotation_amount_recovered_correctly_pct": float(np.mean(recovered[pre] == ss[pre]) * 100),
    "relation_holds_pct_after_first_difference": float(np.mean(consistent[first_diff:]) * 100),
    "relation_holds_pct_random_control": float(ctrl * 100)}
print("E2", res["E2_related_plaintext"])

# ------------------------------------------------------------------ E3 frame length: latency, throughput, packet loss
sr, ch = 44100, 2
music = np.stack([synth_music(30, seed=3), synth_music(30, seed=4)], 1).reshape(-1)    # 30 s stereo
mb = music.size * 2 / 1e6
frames = {"20 ms": int(0.02 * sr) * ch, "100 ms": int(0.1 * sr) * ch, "1 s": sr * ch, "whole file": None}
def bench(fn, reps=5):
    fn(); t = []
    for _ in range(reps):
        s = time.perf_counter(); fn(); t.append(time.perf_counter() - s)
    return float(np.median(t))
e3 = {}
for name, L in frames.items():
    te = bench(lambda: encrypt_audio(KEY, music, sr, ch, chunk=L, nonce=NONCE))
    blob = encrypt_audio(KEY, music, sr, ch, chunk=L, nonce=NONCE); td = bench(lambda: decrypt_audio(KEY, blob))
    nfr = len(blob["chunks"]); per_frame_ms = te / nfr * 1000
    # packet loss: each frame lost independently; lost frames are concealed with silence
    loss = {}
    for pl in (0.01, 0.05):
        kept = rng.random(nfr) >= pl
        out = np.zeros_like(music); Lr = L or music.size
        for k in np.where(kept)[0]:                  # only received frames are verified and decrypted
            out[k * Lr:(k + 1) * Lr] = decrypt_chunk(KEY, blob["nonce"], blob["header"], blob["chunks"][k], blob["tags"][k], int(k), bool(k == nfr - 1))
        recovered = float(np.mean(kept) * 100)
        err = music.astype(float) - out.astype(float)
        snr = float(10 * np.log10((music.astype(float) ** 2).sum() / max((err ** 2).sum(), 1e-9)))
        loss[f"{int(pl * 100)}%"] = {"audio_recovered_pct": recovered, "snr_db": snr}
    # bit errors: probability that a frame survives a binary symmetric channel (analytic)
    bits = 2 * (L or music.size) * 8
    ber = {f"{b:g}": float((1 - b) ** bits * 100) for b in (1e-7, 1e-6, 1e-5)}
    e3[name] = {"frames": nfr, "enc_MBps": mb / te, "dec_MBps": mb / td, "per_frame_ms": per_frame_ms,
                "frame_audio_ms": (L or music.size) / ch / sr * 1000, "packet_loss": loss, "frame_survival_pct_at_ber": ber}
    print("E3", name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in e3[name].items() if k not in ("packet_loss", "frame_survival_pct_at_ber")}, loss)
res["E3_frames"] = {"audio": "30 s, 44.1 kHz stereo", "MB": mb, "results": e3}
# permutation cost: throughput without stage 1 (diffusion + mask only), 1 s frames
hdr1 = make_header(sr, ch, music.size, sr * ch)
def no_perm():
    for k in range(0, music.size, sr * ch):
        encrypt_chunk(KEY, NONCE, hdr1, music[k:k + sr * ch], k // (sr * ch), False, stages=("diff", "mask"))
res["E3_no_permutation_1s_MBps"] = mb / bench(no_perm)
print("E3 no-permutation 1 s frames MB/s", round(res["E3_no_permutation_1s_MBps"], 1))

# ------------------------------------------------------------------ E4 digital silence
z = np.zeros(441000, np.int16)
cz = cipher_samples(encrypt_audio(KEY, z, 44100, 1, nonce=NONCE))
def ent(v, bits):
    c = np.bincount(v, minlength=1 << bits).astype(float); p = c[c > 0] / c.sum(); return float(-(p * np.log2(p)).sum())
bz = np.frombuffer(cz.astype("<u2").tobytes(), np.uint8)
cnt = np.bincount(bz, minlength=256); e = cnt.sum() / 256
s16 = ((cz + 32768) & 0xFFFF) - 32768
res["E4_silence"] = {"samples": int(z.size), "H8": ent(bz, 8), "H16": ent(cz, 16), "H16_ideal": ent(rng.integers(0, 65536, z.size), 16),
                     "chi2": float(((cnt - e) ** 2 / e).sum()), "adj_corr": float(np.corrcoef(s16[:-1], s16[1:])[0, 1])}
print("E4", res["E4_silence"])
json.dump(res, open(os.path.join(args.out, "strengthening_results.json"), "w"), indent=1)
