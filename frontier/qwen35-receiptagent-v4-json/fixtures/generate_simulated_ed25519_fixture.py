"""Print deterministic SIMULATED test signatures, never owner admission.

The already-public RFC 8032 section 7.1 TEST 1 seed is deliberately untrusted.
No key is generated, no credential is read, and no file is written. This fixed
test-only generator has no arguments or interface for signing other payloads.
Production signed_admission.py imports none of this test code.
"""
import hashlib
import json
from pathlib import Path
import sys

from cryptography import __version__
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_signed_admission import canonical, synthetic_payload

PUBLIC_RFC_TEST_SEED = bytes.fromhex(
    "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
RFC_PUBLIC_KEY = "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
RFC_EMPTY_SIGNATURE = (
    "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
    "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b")
TEST_DOMAIN = b"SZL-RECEIPTAGENT-V4-SUPERVISED-GPU-SMOKE-V1\x00"


def main():
    if __version__ != "50.0.0":
        raise RuntimeError("requires the declared test dependency version")
    key = Ed25519PrivateKey.from_private_bytes(PUBLIC_RFC_TEST_SEED)
    if key.public_key().public_bytes_raw().hex() != RFC_PUBLIC_KEY:
        raise RuntimeError("RFC public test key mismatch")
    if key.sign(b"").hex() != RFC_EMPTY_SIGNATURE:
        raise RuntimeError("RFC published primitive vector mismatch")
    payload = canonical(synthetic_payload())
    messages = {
        "correct": TEST_DOMAIN + payload,
        "without_domain": payload,
        "missing_nul": TEST_DOMAIN[:-1] + payload,
        "doubled_nul": TEST_DOMAIN + b"\x00" + payload,
        "wrong_domain": b"SIMULATED-WRONG-DOMAIN\x00" + payload,
        "prehashed": hashlib.sha256(TEST_DOMAIN + payload).digest(),
    }
    print(json.dumps({
        "evidence_class": "SIMULATED",
        "scope": "PUBLIC_RFC_TEST_KEY_NOT_OWNER_AUTHORIZATION",
        "key_id": "SIMULATED",
        "key_source": "https://www.rfc-editor.org/rfc/rfc8032#section-7.1",
        "cryptography_version": __version__,
        "public_key_hex": RFC_PUBLIC_KEY,
        "rfc_empty_signature_hex": RFC_EMPTY_SIGNATURE,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "message_sha256": hashlib.sha256(messages["correct"]).hexdigest(),
        "signatures": {name: key.sign(message).hex() for name, message in messages.items()},
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
