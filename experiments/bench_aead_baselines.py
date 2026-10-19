"""Throughput of standard AEADs, including the misuse-resistant AES-GCM-SIV, against XC-PDA (1 s frames).
Usage: python experiments/bench_aead_baselines.py run1   (repeat with run2, run3, ... and report medians)
Software-AES condition (no AES instructions), e.g. to approximate a Raspberry Pi 4 / Cortex-A72:
    OPENSSL_ia32cap="~0x200000200000000:~0x60000000000" python experiments/bench_aead_baselines.py noaes_run1
This clears only AES-NI and PCLMULQDQ (CPUID.1:ECX bits 25, 1) and VAES and VPCLMULQDQ (CPUID.7:ECX bits 9, 10); the
second field must be given, otherwise OpenSSL also drops AVX2/AVX-512 and slows ChaCha20. ChaCha20-Poly1305 uses none of these
so its throughput serves as a control that nothing else was disabled. Real AES-less hardware is preferable."""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, AESGCMSIV, ChaCha20Poly1305
from xcpda import encrypt_audio
label = sys.argv[1]
rng = np.random.default_rng(1)
x = rng.integers(-32768, 32768, 44100 * 2 * 30).astype(np.int16)        # 30 s of 44.1 kHz stereo
raw = x.tobytes(); mb = len(raw) / 1e6; K = bytes(range(32)); N = bytes(12)
def bench(fn, reps=5):
    fn(); t = []
    for _ in range(reps):
        s = time.perf_counter(); fn(); t.append(time.perf_counter() - s)
    return mb / float(np.median(t))
r = {"AES-256-GCM": bench(lambda: AESGCM(K).encrypt(N, raw, None)),
     "AES-256-GCM-SIV": bench(lambda: AESGCMSIV(K).encrypt(N, raw, None)),
     "ChaCha20-Poly1305": bench(lambda: ChaCha20Poly1305(K).encrypt(N, raw, None)),
     "XC-PDA, 1 s frames": bench(lambda: encrypt_audio(K, x, 44100, 2, chunk=88200, nonce=bytes(24)))}
r["OPENSSL_ia32cap"] = os.environ.get("OPENSSL_ia32cap", "")
print(label, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()})
out = "results/local/bench_aead_baselines.json"
d = json.load(open(out)) if os.path.exists(out) else {}
d[label] = r; json.dump(d, open(out, "w"), indent=1)
