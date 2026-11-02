"""Adiantum (XChaCha12 + AES-256 + NHPoly1305), the length-preserving tweakable cipher of Crowley and Biggers
(IACR ToSC 2018(4)), as used in Linux for disk encryption on processors without AES instructions.

Pure Python/NumPy reference used only to measure ciphertext statistics; it is verified against the
official test vectors (github.com/google/adiantum, also in Linux crypto/testmgr.h) by tests/test_adiantum.py.
It is NOT constant time and is not meant for protecting data or for speed measurements."""
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

_C = np.array([0x61707865, 0x3320646E, 0x79622D32, 0x6B206574], dtype=np.uint32)


def _rotl(x, n):
    return (x << np.uint32(n)) | (x >> np.uint32(32 - n))


def _rounds(s, rounds):
    """s: (16, B) uint32 array; applies `rounds` ChaCha rounds in place."""
    def qr(a, b, c, d):
        s[a] += s[b]; s[d] ^= s[a]; s[d] = _rotl(s[d], 16)
        s[c] += s[d]; s[b] ^= s[c]; s[b] = _rotl(s[b], 12)
        s[a] += s[b]; s[d] ^= s[a]; s[d] = _rotl(s[d], 8)
        s[c] += s[d]; s[b] ^= s[c]; s[b] = _rotl(s[b], 7)
    for _ in range(rounds // 2):
        qr(0, 4, 8, 12); qr(1, 5, 9, 13); qr(2, 6, 10, 14); qr(3, 7, 11, 15)
        qr(0, 5, 10, 15); qr(1, 6, 11, 12); qr(2, 7, 8, 13); qr(3, 4, 9, 14)


def hchacha(key, nonce16, rounds=12):
    s = np.zeros((16, 1), np.uint32)
    s[0:4, 0] = _C; s[4:12, 0] = np.frombuffer(key, "<u4"); s[12:16, 0] = np.frombuffer(nonce16, "<u4")
    _rounds(s, rounds)
    return np.concatenate([s[0:4, 0], s[12:16, 0]]).astype("<u4").tobytes()


def xchacha_stream(key, iv32, nbytes, rounds=12):
    """Linux xchacha: subkey = HChaCha(key, iv[0:16]); state[12..13] = 64-bit counter from iv[24:32],
    state[14..15] = iv[16:24]."""
    sub = np.frombuffer(hchacha(key, iv32[:16], rounds), "<u4")
    nblk = (nbytes + 63) // 64
    ctr0 = int.from_bytes(iv32[24:32], "little")
    ctr = (np.arange(nblk, dtype=np.uint64) + np.uint64(ctr0))
    s = np.zeros((16, nblk), np.uint32)
    s[0:4] = _C[:, None]; s[4:12] = sub[:, None]
    s[12] = (ctr & np.uint64(0xFFFFFFFF)).astype(np.uint32); s[13] = (ctr >> np.uint64(32)).astype(np.uint32)
    s[14:16] = np.frombuffer(iv32[16:24], "<u4")[:, None]
    x = s.copy(); _rounds(x, rounds); x += s
    return x.T.astype("<u4").tobytes()[:nbytes]


_P = (1 << 130) - 5


def _poly_core(rkey16, data):
    """Poly1305 'core' as in Linux: clamped r, full blocks with the 2^128 bit, no final addition of s."""
    r = int.from_bytes(rkey16, "little") & 0x0FFFFFFC0FFFFFFC0FFFFFFC0FFFFFFF
    h = 0
    for i in range(0, len(data), 16):
        h = (h + int.from_bytes(data[i:i + 16], "little") + (1 << 128)) * r % _P
    return h % (1 << 128)


NH_KEY_BYTES = (256 + 2 * 2 * 3) * 4


def _nhpoly1305(key, msg):
    pkey, nkey = key[:16], np.frombuffer(key[16:16 + NH_KEY_BYTES], "<u4").astype(np.uint64)
    if len(msg) % 16:
        msg = msg + bytes(16 - len(msg) % 16)
    digests = bytearray()
    for off in range(0, len(msg), 1024):
        m = np.frombuffer(msg[off:off + 1024], "<u4").astype(np.uint64).reshape(-1, 4)
        u = m.shape[0]; out = []
        for p in range(4):
            k = nkey[4 * p: 4 * p + 4 * u].reshape(-1, 4)
            a = ((m[:, 0] + k[:, 0]) & 0xFFFFFFFF) * ((m[:, 2] + k[:, 2]) & 0xFFFFFFFF)
            b = ((m[:, 1] + k[:, 1]) & 0xFFFFFFFF) * ((m[:, 3] + k[:, 3]) & 0xFFFFFFFF)
            with np.errstate(over="ignore"):
                out.append(np.uint64(np.sum(a, dtype=np.uint64) + np.sum(b, dtype=np.uint64)))
        digests += np.array(out, dtype="<u8").tobytes()
    return _poly_core(pkey, bytes(digests))


def _subkeys(key):
    d = xchacha_stream(key, b"\x01" + bytes(31), 32 + 16 + 16 + NH_KEY_BYTES)
    return d[:32], d[32:48], d[48:]


def encrypt(key, tweak32, ptext):
    """Adiantum encryption of one message (>= 16 bytes) under a 32-byte key and 32-byte tweak."""
    assert len(ptext) >= 16 and len(tweak32) == 32
    ke, kt, kh = _subkeys(key)
    pl, pr = ptext[:-16], ptext[-16:]
    ht = _poly_core(kt, (len(pl) * 8).to_bytes(8, "little") + bytes(8) + tweak32)
    pm = (int.from_bytes(pr, "little") + _nhpoly1305(kh, pl) + ht) % (1 << 128)
    cm = Cipher(algorithms.AES(ke), modes.ECB()).encryptor().update(pm.to_bytes(16, "little"))
    ks = xchacha_stream(key, cm + (1).to_bytes(4, "little") + bytes(12), len(pl))
    cl = (np.frombuffer(pl, np.uint8) ^ np.frombuffer(ks, np.uint8)).tobytes()
    cr = (int.from_bytes(cm, "little") - _nhpoly1305(kh, cl) - ht) % (1 << 128)
    return cl + cr.to_bytes(16, "little")


def decrypt(key, tweak32, ctext):
    ke, kt, kh = _subkeys(key)
    cl, cr = ctext[:-16], ctext[-16:]
    ht = _poly_core(kt, (len(cl) * 8).to_bytes(8, "little") + bytes(8) + tweak32)
    cm = (int.from_bytes(cr, "little") + _nhpoly1305(kh, cl) + ht) % (1 << 128)
    cmb = cm.to_bytes(16, "little")
    ks = xchacha_stream(key, cmb + (1).to_bytes(4, "little") + bytes(12), len(cl))
    pl = (np.frombuffer(cl, np.uint8) ^ np.frombuffer(ks, np.uint8)).tobytes()
    pm = Cipher(algorithms.AES(ke), modes.ECB()).decryptor().update(cmb)
    pr = (int.from_bytes(pm, "little") - _nhpoly1305(kh, pl) - ht) % (1 << 128)
    return pl + pr.to_bytes(16, "little")
