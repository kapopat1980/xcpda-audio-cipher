"""Command-line use:
    python -m xcpda keygen  key.bin
    python -m xcpda encrypt input.wav  output.xcpda --key key.bin [--frame-seconds 1.0]
    python -m xcpda decrypt input.xcpda output.wav  --key key.bin
Input WAV files must be 16-bit PCM (mono or multichannel)."""
import argparse, os, struct, sys
import numpy as np
from scipy.io import wavfile
from .cipher import encrypt_audio, decrypt_audio, pack, unpack, AuthError


def _key(path):
    k = open(path, "rb").read()
    if len(k) != 32:
        sys.exit("key file must contain exactly 32 bytes")
    return k


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m xcpda", description="XC-PDA authenticated audio encryption")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("keygen"); g.add_argument("key")
    e = sub.add_parser("encrypt"); e.add_argument("input"); e.add_argument("output"); e.add_argument("--key", required=True)
    e.add_argument("--frame-seconds", type=float, default=0.0, help="0 = whole file as one frame")
    d = sub.add_parser("decrypt"); d.add_argument("input"); d.add_argument("output"); d.add_argument("--key", required=True)
    a = p.parse_args(argv)
    if a.cmd == "keygen":
        with open(a.key, "wb") as f: f.write(os.urandom(32))
        os.chmod(a.key, 0o600); print(f"wrote 256-bit key to {a.key}")
    elif a.cmd == "encrypt":
        sr, x = wavfile.read(a.input)
        if x.dtype != np.int16:
            sys.exit("input must be 16-bit PCM WAV")
        ch = 1 if x.ndim == 1 else x.shape[1]
        chunk = int(round(a.frame_seconds * sr)) * ch if a.frame_seconds > 0 else None
        blob = encrypt_audio(_key(a.key), x.reshape(-1), sr, ch, chunk=chunk)
        with open(a.output, "wb") as f: f.write(pack(blob))
        print(f"encrypted {x.size} samples ({ch} ch, {sr} Hz) in {len(blob['chunks'])} frame(s) -> {a.output}")
    else:
        try:
            blob = unpack(open(a.input, "rb").read())
            y = decrypt_audio(_key(a.key), blob)
        except AuthError:
            sys.exit("authentication failed: wrong key or modified file")
        sr, ch = struct.unpack("<IH", blob["header"][:6])
        wavfile.write(a.output, sr, y.reshape(-1, ch) if ch > 1 else y)
        print(f"decrypted and verified -> {a.output}")


if __name__ == "__main__":
    main()
