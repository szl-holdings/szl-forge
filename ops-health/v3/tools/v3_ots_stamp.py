"""Stamp files with OpenTimestamps using the core library only (prereg 3.5 fallback path).

The reference CLI (opentimestamps-client 0.7.2) fails on Windows because python-bitcoinlib's
key module asks ctypes for an OpenSSL library that is not found. Stamping does not need that
module, so this script uses `opentimestamps` (python-opentimestamps 0.4.5) directly:

- digest = sha256(file bytes); for the digest-input files of v3, that is the registered digest M;
- each calendar receives only the 32-byte digest (RemoteCalendar.submit -> POST <cal>/digest);
- every calendar's returned timestamp is merged under the file digest;
- the result is written as a standard detached proof `<file>.ots`, readable by `ots info`.

Usage: python scripts/v3_ots_stamp.py <min_calendars> <file> [<file> ...]
"""
import hashlib
import sys

from opentimestamps.calendar import RemoteCalendar
from opentimestamps.core.op import OpSHA256
from opentimestamps.core.serialize import StreamSerializationContext
from opentimestamps.core.timestamp import DetachedTimestampFile, Timestamp

CALENDARS = (
    "https://a.pool.opentimestamps.org",
    "https://b.pool.opentimestamps.org",
    "https://a.pool.eternitywall.com",
    "https://ots.btc.catallaxy.com",
)
GROUP_A = {"https://a.pool.opentimestamps.org", "https://b.pool.opentimestamps.org"}
GROUP_B = {"https://a.pool.eternitywall.com", "https://ots.btc.catallaxy.com"}


def stamp(path, minimum):
    data = open(path, "rb").read()
    digest = hashlib.sha256(data).digest()
    detached = DetachedTimestampFile(OpSHA256(), Timestamp(digest))
    ok = []
    for url in CALENDARS:
        try:
            calendar_ts = RemoteCalendar(url, user_agent="szl-oac-v3-stamp/1.0").submit(digest, timeout=30)
            detached.timestamp.merge(calendar_ts)
            ok.append(url)
            print(f"  {url}: ok")
        except Exception as exc:  # noqa: BLE001 -- record and continue with the other calendars
            print(f"  {url}: FAILED {type(exc).__name__}: {exc}")
    groups_ok = bool(GROUP_A & set(ok)) and bool(GROUP_B & set(ok))
    if len(ok) < minimum or not groups_ok:
        raise SystemExit(f"{path}: only {len(ok)} calendars ({ok}); need >= {minimum} and one per group")
    with open(path + ".ots", "wb") as handle:
        detached.serialize(StreamSerializationContext(handle))
    print(f"{path}: digest {digest.hex()} -> {path}.ots ({len(ok)} calendars)")


if __name__ == "__main__":
    minimum = int(sys.argv[1])
    for name in sys.argv[2:]:
        stamp(name, minimum)
