import os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from xcpda import encrypt_audio, decrypt_audio, cipher_samples, pack, unpack, AuthError

rng = np.random.default_rng(0)
KEY = bytes(range(32))


@pytest.mark.parametrize("n,chunk", [(1, None), (7, None), (50_000, None), (50_000, 11_025), (100_001, 30_000)])
def test_roundtrip(n, chunk):
    x = rng.integers(-32768, 32768, n).astype(np.int16)
    assert np.array_equal(decrypt_audio(KEY, encrypt_audio(KEY, x, 44100, 1, chunk=chunk)), x)


def test_file_format_roundtrip():
    x = rng.integers(-32768, 32768, 30_000).astype(np.int16)
    blob = unpack(pack(encrypt_audio(KEY, x, 16000, 2, chunk=8000)))
    assert np.array_equal(decrypt_audio(KEY, blob), x)


def test_wrong_key_rejected():
    x = rng.integers(-32768, 32768, 5_000).astype(np.int16)
    blob = encrypt_audio(KEY, x, 16000, 1)
    with pytest.raises(AuthError):
        decrypt_audio(bytes(32), blob)


def test_bit_flip_rejected():
    x = rng.integers(-32768, 32768, 5_000).astype(np.int16)
    blob = encrypt_audio(KEY, x, 16000, 1)
    c = bytearray(blob["chunks"][0]); c[100] ^= 1; blob["chunks"][0] = bytes(c)
    with pytest.raises(AuthError):
        decrypt_audio(KEY, blob)


def test_frame_swap_and_truncation_rejected():
    x = rng.integers(-32768, 32768, 40_000).astype(np.int16)
    blob = encrypt_audio(KEY, x, 16000, 1, chunk=10_000)
    swapped = dict(blob, chunks=[blob["chunks"][1], blob["chunks"][0]] + blob["chunks"][2:],
                   tags=[blob["tags"][1], blob["tags"][0]] + blob["tags"][2:])
    with pytest.raises(AuthError):
        decrypt_audio(KEY, swapped)
    with pytest.raises(AuthError):
        decrypt_audio(KEY, dict(blob, chunks=blob["chunks"][:-1], tags=blob["tags"][:-1]))
    with pytest.raises(AuthError):
        unpack(pack(blob)[:-20])


def test_plaintext_sensitivity():
    x = rng.integers(-32768, 32768, 20_000).astype(np.int16)
    nonce = bytes(24)
    c1 = cipher_samples(encrypt_audio(KEY, x, 16000, 1, nonce=nonce))
    x2 = x.copy(); x2[1234] ^= 1
    c2 = cipher_samples(encrypt_audio(KEY, x2, 16000, 1, nonce=nonce))
    assert np.mean(c1 != c2) > 0.999


def test_nonce_reuse_does_not_leak_xor():
    a = rng.integers(-32768, 32768, 20_000).astype(np.int16)
    b = rng.integers(-32768, 32768, 20_000).astype(np.int16)
    nonce = bytes(24)
    ca = cipher_samples(encrypt_audio(KEY, a, 16000, 1, nonce=nonce))
    cb = cipher_samples(encrypt_audio(KEY, b, 16000, 1, nonce=nonce))
    pa, pb = a.astype(np.int64) & 0xFFFF, b.astype(np.int64) & 0xFFFF
    assert np.mean((ca ^ cb) == (pa ^ pb)) < 0.001


def test_documented_related_plaintext_relation():
    """Documents the known limitation (paper, Section 6.6): under a repeated nonce, frames that share the forward-chain
    prefix satisfy dW_i[s_i] == dW_{i+1}[0]. If a future version removes it, this test should be inverted."""
    from xcpda import encrypt_chunk, make_header
    from xcpda.cipher import derive_subkeys, _diff_material, chacha_stream, _perm_indices
    N, nonce = 1024, bytes(24); hdr = make_header(16000, 1, N, N)
    base = rng.integers(-32768, 32768, N).astype(np.int16); x2 = base.copy(); x2[N - 1] ^= 1
    d = np.frombuffer(encrypt_chunk(KEY, nonce, hdr, base)[0], "<u2").astype(np.int64) ^ \
        np.frombuffer(encrypt_chunk(KEY, nonce, hdr, x2)[0], "<u2").astype(np.int64)
    kp, kd, _, _ = derive_subkeys(KEY, nonce, hdr, 0, True)
    ss = _diff_material(kd, N)[3]
    k = int(np.argsort(_perm_indices(N, np.frombuffer(chacha_stream(kp, 8 * N, 1), "<u8")))[N - 1])
    assert all(((d[i] >> ss[i]) & 1) == (d[i + 1] & 1) for i in range(k - 1))
