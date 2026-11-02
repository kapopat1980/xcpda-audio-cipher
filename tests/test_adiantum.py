"""Check the reference Adiantum implementation against the official test vectors (Linux crypto/testmgr.h)."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))
import adiantum

V = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectors", "adiantum_linux_testmgr.json")))


def test_adiantum_xchacha12_aes_vectors():
    for v in V["adiantum_xchacha12_aes_tv_template"]:
        k, iv, p, c = (bytes.fromhex(v[f]) for f in ("key", "iv", "ptext", "ctext"))
        assert adiantum.encrypt(k, iv, p) == c
        assert adiantum.decrypt(k, iv, c) == p
