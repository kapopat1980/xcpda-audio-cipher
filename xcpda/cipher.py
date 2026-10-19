"""
XC-PDA: eXtended-nonce ChaCha20 Permutation-Diffusion Authenticated cipher for PCM audio.

Encryption of one frame (chunk) of 16-bit PCM samples P[0..N-1]:
  0. Subkeys   : (K_perm, K_diff, K_mask, K_mac) = HKDF-SHA256(K, salt = nonce192,
                 info = "XC-PDA/v1" || header || chunk_index || last_flag)
  1. Permute   : Fisher-Yates shuffle of sample indices driven by ChaCha20(K_perm)
  2. Diffuse   : forward chain  u_i = ROTL16((x_i ^ a_i) + u_{i-1}, r_i)
                 backward chain w_i = ROTL16((u_i ^ b_i) + w_{i+1}, s_i)
                 (a, b, r, s, IVs taken from ChaCha20(K_diff))
  3. Mask      : C = W ^ ChaCha20(K_mask)
  4. Tag       : T = Poly1305(K_mac, header || nonce || chunk_index || last || C)
Decryption verifies T first, then inverts 3 -> 2 -> 1.
"""
import os, struct, hmac
import numpy as np
from numba import njit
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes, poly1305

MAGIC = b"XC-PDA/v1"
ZERO16 = b"\x00" * 16


def chacha_stream(key: bytes, nbytes: int, stream_id: int) -> bytes:
    # 16-byte nonce for `cryptography` = 4-byte LE counter || 12-byte nonce; stream_id separates domains
    nonce = struct.pack("<I", 0) + struct.pack("<Q", stream_id) + b"\x00\x00\x00\x00"
    enc = Cipher(algorithms.ChaCha20(key, nonce), mode=None).encryptor()
    return enc.update(b"\x00" * nbytes)


def derive_subkeys(key: bytes, nonce: bytes, header: bytes, chunk_idx: int, last: bool):
    info = MAGIC + header + struct.pack("<QB", chunk_idx, 1 if last else 0)
    okm = HKDF(algorithm=hashes.SHA256(), length=128, salt=nonce, info=info).derive(key)
    return okm[0:32], okm[32:64], okm[64:96], okm[96:128]


@njit(cache=True)
def _perm_indices(n, rnd):
    idx = np.arange(n)
    for i in range(n - 1, 0, -1):
        j = rnd[i] % np.uint64(i + 1)
        t = idx[i]; idx[i] = idx[j]; idx[j] = t
    return idx


@njit(cache=True)
def _rotl16(x, r):
    r &= 15
    return ((x << r) | (x >> ((16 - r) & 15))) & 0xFFFF if r else x


@njit(cache=True)
def _rotr16(x, r):
    r &= 15
    return ((x >> r) | (x << ((16 - r) & 15))) & 0xFFFF if r else x


@njit(cache=True)
def _diffuse_fwd(x, a, b, rr, ss, iv1, iv2):
    n = x.shape[0]
    u = np.empty(n, np.int64)
    prev = iv1
    for i in range(n):
        prev = _rotl16(((x[i] ^ a[i]) + prev) & 0xFFFF, rr[i])
        u[i] = prev
    w = np.empty(n, np.int64)
    nxt = iv2
    for i in range(n - 1, -1, -1):
        nxt = _rotl16(((u[i] ^ b[i]) + nxt) & 0xFFFF, ss[i])
        w[i] = nxt
    return w


@njit(cache=True)
def _diffuse_inv(w, a, b, rr, ss, iv1, iv2):
    n = w.shape[0]
    u = np.empty(n, np.int64)
    for i in range(n):  # every w_{i+1} is known, so this inverse is position-independent
        nxt = w[i + 1] if i + 1 < n else iv2
        u[i] = ((_rotr16(w[i], ss[i]) - nxt) & 0xFFFF) ^ b[i]
    x = np.empty(n, np.int64)
    for i in range(n):
        prev = u[i - 1] if i > 0 else iv1
        x[i] = ((_rotr16(u[i], rr[i]) - prev) & 0xFFFF) ^ a[i]
    return x


def _diff_material(k_diff, n):
    raw = np.frombuffer(chacha_stream(k_diff, 5 * n + 4, 2), dtype=np.uint8)
    a = raw[0:2 * n].view("<u2").astype(np.int64)
    b = raw[2 * n:4 * n].view("<u2").astype(np.int64)
    rs = raw[4 * n:5 * n].astype(np.int64)
    rr, ss = rs & 15, rs >> 4
    iv1 = int.from_bytes(bytes(raw[5 * n:5 * n + 2]), "little")
    iv2 = int.from_bytes(bytes(raw[5 * n + 2:5 * n + 4]), "little")
    return a, b, rr, ss, iv1, iv2


def _tag(k_mac, header, nonce, chunk_idx, last, cbytes):
    p = poly1305.Poly1305(k_mac)
    p.update(MAGIC + header + nonce + struct.pack("<QBQ", chunk_idx, 1 if last else 0, len(cbytes)) + cbytes)
    return p.finalize()


def encrypt_chunk(key, nonce, header, samples_i16, chunk_idx=0, last=True, stages=("perm", "diff", "mask"), timings=None):
    import time
    kp, kd, km, kmac = derive_subkeys(key, nonce, header, chunk_idx, last)
    x = samples_i16.astype(np.int64) & 0xFFFF
    n = x.shape[0]
    t0 = time.perf_counter()
    if "perm" in stages:
        rnd = np.frombuffer(chacha_stream(kp, 8 * n, 1), dtype="<u8")
        perm = _perm_indices(n, rnd)
        x = x[perm]
    t1 = time.perf_counter()
    if "diff" in stages:
        x = _diffuse_fwd(x, *_diff_material(kd, n))
    t2 = time.perf_counter()
    if "mask" in stages:
        ks = np.frombuffer(chacha_stream(km, 2 * n, 3), dtype="<u2").astype(np.int64)
        x = x ^ ks
    c = (x & 0xFFFF).astype("<u2").tobytes()
    t3 = time.perf_counter()
    tag = _tag(kmac, header, nonce, chunk_idx, last, c)
    t4 = time.perf_counter()
    if timings is not None:
        for k, v in zip(("perm", "diff", "mask", "tag"), (t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
            timings[k] = timings.get(k, 0.0) + v
    return c, tag


class AuthError(Exception):
    pass


def decrypt_chunk(key, nonce, header, c, tag, chunk_idx=0, last=True, verify=True):
    kp, kd, km, kmac = derive_subkeys(key, nonce, header, chunk_idx, last)
    if verify and not hmac.compare_digest(_tag(kmac, header, nonce, chunk_idx, last, c), tag):
        raise AuthError("tag mismatch")
    y = np.frombuffer(c, dtype="<u2").astype(np.int64)
    n = y.shape[0]
    y = y ^ np.frombuffer(chacha_stream(km, 2 * n, 3), dtype="<u2").astype(np.int64)
    y = _diffuse_inv(y, *_diff_material(kd, n))
    rnd = np.frombuffer(chacha_stream(kp, 8 * n, 1), dtype="<u8")
    perm = _perm_indices(n, rnd)
    x = np.empty(n, np.int64)
    x[perm] = y
    return (x & 0xFFFF).astype(np.uint16).view(np.int16)


def make_header(sr, ch, n_samples, chunk):
    return struct.pack("<IHHQI", sr, ch, 16, n_samples, chunk)


def encrypt_audio(key, samples_i16, sr, ch, chunk=None, nonce=None, timings=None):
    """samples_i16: 1-D interleaved int16. chunk = samples per frame (None = whole file)."""
    n = samples_i16.shape[0]
    chunk = chunk or n
    nonce = nonce or os.urandom(24)
    header = make_header(sr, ch, n, chunk)
    out, tags = [], []
    nchunks = (n + chunk - 1) // chunk
    for k in range(nchunks):
        seg = samples_i16[k * chunk:(k + 1) * chunk]
        c, t = encrypt_chunk(key, nonce, header, seg, k, k == nchunks - 1, timings=timings)
        out.append(c); tags.append(t)
    return {"header": header, "nonce": nonce, "chunks": out, "tags": tags}


def decrypt_audio(key, blob, verify=True):
    sr, ch, bits, n, chunk = struct.unpack("<IHHQI", blob["header"])
    parts = []
    m = len(blob["chunks"])
    for k, (c, t) in enumerate(zip(blob["chunks"], blob["tags"])):
        parts.append(decrypt_chunk(key, blob["nonce"], blob["header"], c, t, k, k == m - 1, verify))
    return np.concatenate(parts)


def cipher_samples(blob):
    return np.frombuffer(b"".join(blob["chunks"]), dtype="<u2").astype(np.int64)


# ---------------------------------------------------------------------------------
# Container format (.xcpda):  b"XCPDA1" || header (20 B) || nonce (24 B)
#                             || for each frame: ciphertext (2 * frame_len B) || tag (16 B)
# Frame lengths follow from the header (total samples, frame length), so no extra framing is needed.
# ---------------------------------------------------------------------------------
FILE_MAGIC = b"XCPDA1"


def pack(blob) -> bytes:
    out = [FILE_MAGIC, blob["header"], blob["nonce"]]
    for c, t in zip(blob["chunks"], blob["tags"]):
        out += [c, t]
    return b"".join(out)


def unpack(data: bytes):
    if data[:6] != FILE_MAGIC:
        raise ValueError("not an XC-PDA file")
    header, nonce, pos = data[6:26], data[26:50], 50
    sr, ch, bits, n, chunk = struct.unpack("<IHHQI", header)
    chunks, tags = [], []
    remaining = n
    while remaining > 0:
        m = min(chunk, remaining)
        chunks.append(data[pos:pos + 2 * m]); pos += 2 * m
        tags.append(data[pos:pos + 16]); pos += 16
        remaining -= m
    if pos != len(data) or any(len(t) != 16 for t in tags):
        raise AuthError("truncated or extended file")
    return {"header": header, "nonce": nonce, "chunks": chunks, "tags": tags}
