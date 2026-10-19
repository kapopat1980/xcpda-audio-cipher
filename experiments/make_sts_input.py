"""Build 1,000 x 10^6-bit NIST STS input from XC-PDA ciphertext: each recording (LibreOffice gallery clips with random gain
and noise, or 10 s of digital silence) is encrypted under its own random key. Usage: make_sts_input.py audio|silence out.bin"""
import sys, os, glob, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scipy.io import wavfile
from xcpda import encrypt_audio
mode, outpath = sys.argv[1], sys.argv[2]; target = 125000 * 1000
rng = np.random.default_rng(7)
clips = []
for f in sorted(glob.glob(os.environ.get("XC_GALLERY", "/usr/lib/libreoffice/share/gallery/sounds") + "/*.wav")):
    sr, x = wavfile.read(f)
    if x.dtype == np.int16 and x.ndim == 1: clips.append((sr, x))
out = bytearray()
i = 0
while len(out) < target:
    if mode == "silence":
        sr, x = 44100, np.zeros(441000, np.int16)                     # 10 s of digital silence
    else:
        sr, x = clips[i % len(clips)]
        if i >= len(clips):                                            # vary the audio: gain and offset
            g = rng.uniform(0.2, 1.0); x = np.clip(x.astype(float) * g + rng.normal(0, 50, x.size), -32768, 32767).astype(np.int16)
    out += b"".join(encrypt_audio(os.urandom(32), x, sr, 1)["chunks"])
    i += 1
open(outpath, "wb").write(bytes(out[:target]))
print(mode, i, "files encrypted,", len(out[:target]), "bytes")
