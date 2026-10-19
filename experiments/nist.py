"""Subset of NIST SP 800-22 rev1a statistical tests (vectorised numpy)."""
import numpy as np
from scipy.special import erfc, gammaincc
from scipy.stats import norm


def monobit(b):
    n = b.size
    s = np.sum(2 * b.astype(np.int64) - 1)
    return erfc(abs(s) / np.sqrt(n) / np.sqrt(2))


def block_frequency(b, M=128):
    N = b.size // M
    pi = b[:N * M].reshape(N, M).mean(axis=1)
    chi2 = 4 * M * np.sum((pi - 0.5) ** 2)
    return gammaincc(N / 2, chi2 / 2)


def runs(b):
    n = b.size
    pi = b.mean()
    if abs(pi - 0.5) >= 2 / np.sqrt(n):
        return 0.0
    V = 1 + np.count_nonzero(b[1:] != b[:-1])
    return erfc(abs(V - 2 * n * pi * (1 - pi)) / (2 * np.sqrt(2 * n) * pi * (1 - pi)))


def longest_run(b):
    M, K = 10000, 6
    N = b.size // M
    pis = np.array([0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727])
    v = np.zeros(7)
    for blk in b[:N * M].reshape(N, M):
        x = np.concatenate(([0], blk.astype(np.int8), [0]))
        d = np.diff(x)
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        L = (ends - starts).max() if starts.size else 0
        v[min(max(L, 10), 16) - 10] += 1
    chi2 = np.sum((v - N * pis) ** 2 / (N * pis))
    return gammaincc(K / 2, chi2 / 2)


def dft(b):
    n = b.size
    X = 2 * b.astype(np.float64) - 1
    m = np.abs(np.fft.rfft(X))[: n // 2]
    T = np.sqrt(np.log(1 / 0.05) * n)
    N0 = 0.95 * n / 2
    N1 = np.count_nonzero(m < T)
    d = (N1 - N0) / np.sqrt(n * 0.95 * 0.05 / 4)
    return erfc(abs(d) / np.sqrt(2))


def _pattern_counts(b, m):
    n = b.size
    if m == 0:
        return np.array([n])
    ext = np.concatenate((b, b[: m - 1])).astype(np.int64)
    val = np.zeros(n, np.int64)
    for k in range(m):
        val = (val << 1) | ext[k:k + n]
    return np.bincount(val, minlength=1 << m)


def approximate_entropy(b, m=10):
    n = b.size

    def phi(mm):
        c = _pattern_counts(b, mm) / n
        c = c[c > 0]
        return np.sum(c * np.log(c))
    apen = phi(m) - phi(m + 1)
    chi2 = 2 * n * (np.log(2) - apen)
    return gammaincc(2 ** (m - 1), chi2 / 2)


def serial(b, m=16):
    n = b.size

    def psi(mm):
        if mm <= 0:
            return 0.0
        v = _pattern_counts(b, mm).astype(np.float64)
        return (2 ** mm / n) * np.sum(v ** 2) - n
    p0, p1, p2 = psi(m), psi(m - 1), psi(m - 2)
    d1 = p0 - p1
    d2 = p0 - 2 * p1 + p2
    return gammaincc(2 ** (m - 2), d1 / 2), gammaincc(2 ** (m - 3), d2 / 2)


def cusum(b):
    n = b.size
    S = np.cumsum(2 * b.astype(np.int64) - 1)
    z = np.max(np.abs(S))
    k1 = np.arange(int((-n / z + 1) / 4), int((n / z - 1) / 4) + 1)
    k2 = np.arange(int((-n / z - 3) / 4), int((n / z - 1) / 4) + 1)
    sq = np.sqrt(n)
    p = 1 - np.sum(norm.cdf((4 * k1 + 1) * z / sq) - norm.cdf((4 * k1 - 1) * z / sq)) \
        + np.sum(norm.cdf((4 * k2 + 3) * z / sq) - norm.cdf((4 * k2 + 1) * z / sq))
    return float(p)


def run_all(b):
    s1, s2 = serial(b)
    return {
        "Frequency (monobit)": monobit(b),
        "Block frequency (M=128)": block_frequency(b),
        "Runs": runs(b),
        "Longest run of ones (M=10^4)": longest_run(b),
        "Discrete Fourier transform": dft(b),
        "Approximate entropy (m=10)": approximate_entropy(b),
        "Serial-1 (m=16)": s1,
        "Serial-2 (m=16)": s2,
        "Cumulative sums (forward)": cusum(b),
    }
