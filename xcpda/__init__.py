"""XC-PDA: a ChaCha20-based permutation-diffusion authenticated cipher for PCM audio."""
from .cipher import (encrypt_audio, decrypt_audio, encrypt_chunk, decrypt_chunk, cipher_samples,
                     make_header, derive_subkeys, pack, unpack, AuthError)

__version__ = "1.0.0"
__all__ = ["encrypt_audio", "decrypt_audio", "encrypt_chunk", "decrypt_chunk", "cipher_samples",
           "make_header", "derive_subkeys", "pack", "unpack", "AuthError"]
