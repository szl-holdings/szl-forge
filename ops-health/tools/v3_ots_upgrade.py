"""Upgrade OAC v3 OpenTimestamps proofs and evaluate T_fwd (prereg section 3.5 cutoff rule).

Usage: python v3_ots_upgrade.py <file.ots> [...]. Writes <file>.ots.upgraded when an upgrade arrives.
T_fwd(H) = median (6th smallest) of header times of blocks H..H+10, from two explorers that must agree.
"""
import datetime, hashlib, json, sys, urllib.request
from opentimestamps.calendar import RemoteCalendar
from opentimestamps.core.notary import BitcoinBlockHeaderAttestation, PendingAttestation
from opentimestamps.core.serialize import StreamDeserializationContext, StreamSerializationContext
from opentimestamps.core.timestamp import DetachedTimestampFile

EXPLORERS = ("https://blockstream.info/api", "https://mempool.space/api")

def walk(ts):
    yield ts
    for sub in ts.ops.values():
        yield from walk(sub)

def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "szl-oac-v3-ots/1.0"}), timeout=30) as r:
        return r.read().decode()

def header(api, height):
    h = get(f"{api}/block-height/{height}").strip()
    b = json.loads(get(f"{api}/block/{h}"))
    return {"hash": h, "merkle_root": b["merkle_root"], "time": b["timestamp"]}

def t_fwd(height):
    out = {}
    for api in EXPLORERS:
        times, first = [], None
        for k in range(11):
            hd = header(api, height + k)
            first = first or hd
            times.append(hd["time"])
        out[api] = {"block_hash": first["hash"], "merkle_root": first["merkle_root"], "times": times, "t_fwd": sorted(times)[5]}
    vals = {json.dumps([v["block_hash"], v["times"]]) for v in out.values()}
    return out, len(vals) == 1

def main(paths):
    report = {"checked_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "files": {}}
    for path in paths:
        with open(path, "rb") as f:
            dtf = DetachedTimestampFile.deserialize(StreamDeserializationContext(f))
        calls = []
        for sub in list(walk(dtf.timestamp)):
            for att in list(sub.attestations):
                if isinstance(att, PendingAttestation):
                    try:
                        sub.merge(RemoteCalendar(att.uri).get_timestamp(sub.msg, timeout=30))
                        calls.append({"calendar": att.uri, "result": "upgraded"})
                    except Exception as exc:  # noqa: BLE001 -- record and continue with the other calendars
                        calls.append({"calendar": att.uri, "result": f"{type(exc).__name__}: {str(exc)[:120]}"})
        btc = []
        for msg, att in dtf.timestamp.all_attestations():
            if isinstance(att, BitcoinBlockHeaderAttestation):
                btc.append({"height": att.height, "attested_merkle_root": msg[::-1].hex()})
        entry = {"file_digest": dtf.file_digest.hex(), "calendar_calls": calls, "bitcoin": []}
        for b in sorted(btc, key=lambda x: x["height"]):
            ex, agree = t_fwd(b["height"])
            roots_ok = all(v["merkle_root"] == b["attested_merkle_root"] for v in ex.values())
            tf = next(iter(ex.values()))["t_fwd"]
            entry["bitcoin"].append({**b, "explorers_agree": agree, "merkle_root_matches": roots_ok, "t_fwd_unix": tf,
                "t_fwd_utc": datetime.datetime.fromtimestamp(tf, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "explorers": ex})
        if btc:
            with open(path + ".upgraded", "wb") as f:
                dtf.serialize(StreamSerializationContext(f))
            entry["upgraded_file_sha256"] = hashlib.sha256(open(path + ".upgraded", "rb").read()).hexdigest()
        report["files"][path] = entry
    print(json.dumps(report, indent=1))

if __name__ == "__main__":
    main(sys.argv[1:])
