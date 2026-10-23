#!/usr/bin/env python3
"""v3 registration: derive the drand seal round R from saved chain-info responses.

Standard library only. Reads the raw /info and /{hash}/info responses and the
/{hash}/public/latest responses saved under logs/v3/register/, checks that every relay
agrees byte-for-byte on the chain parameters, computes R for the registered target
instant, and checks randomness == sha256(signature) and cross-relay agreement on the
latest round (a smoke test of the opening's fallback check; no BLS pairing here).

Usage: python -I scripts/v3_beacon_round.py <register_dir> <target_utc_iso> <out_txt>
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

RELAYS = ("api.drand.sh", "drand.cloudflare.com", "api2.drand.sh")
STAMP = "20261013"
KEYS = ("hash", "public_key", "period", "genesis_time", "schemeID", "groupHash")


def load(path):
    raw = path.read_bytes()
    return raw, json.loads(raw.decode("utf-8"))


def iso(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    reg = Path(sys.argv[1])
    target = datetime.strptime(sys.argv[2], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    out = Path(sys.argv[3])
    lines = []
    infos = {}
    for relay in RELAYS:
        for kind in ("info", "chaininfo"):
            p = reg / f"{kind}_{relay}_{STAMP}.json"
            raw, obj = load(p)
            infos[(relay, kind)] = obj
            lines.append(f"{p.name} sha256={hashlib.sha256(raw).hexdigest()} bytes={len(raw)}")
    ref = infos[(RELAYS[0], "info")]
    agree = all(all(o.get(k) == ref.get(k) for k in KEYS) for o in infos.values())
    lines.append(f"chain parameters identical across {len(infos)} responses: {agree}")
    if not agree:
        raise SystemExit("relay disagreement on chain parameters")
    chain_hash = ref["hash"]
    period = int(ref["period"])
    genesis = int(ref["genesis_time"])
    for k in KEYS:
        lines.append(f"{k} = {ref[k]}")
    lines.append(f"beaconID = {ref.get('metadata', {}).get('beaconID')}")
    lines.append(f"genesis_time UTC = {iso(genesis)}")

    t = int(target.timestamp())
    # round r is scheduled at genesis + (r - 1) * period; R = least r with that time >= t
    r = (t - genesis + period - 1) // period + 1
    assert genesis + (r - 1) * period >= t and genesis + (r - 2) * period < t
    lines.append(f"target = {iso(t)} (unix {t})")
    lines.append(f"R = {r}")
    lines.append(f"R scheduled = {iso(genesis + (r - 1) * period)} (unix {genesis + (r - 1) * period})")
    lines.append(f"R-1 scheduled = {iso(genesis + (r - 2) * period)}")

    latest = {}
    for relay in RELAYS:
        p = reg / f"latest_{relay}_{STAMP}.json"
        raw, obj = load(p)
        latest[relay] = obj
        sig = bytes.fromhex(obj["signature"])
        ok = hashlib.sha256(sig).hexdigest() == obj["randomness"]
        sched = genesis + (int(obj["round"]) - 1) * period
        lines.append(
            f"{p.name} sha256={hashlib.sha256(raw).hexdigest()} round={obj['round']} "
            f"scheduled={iso(sched)} randomness==sha256(signature): {ok}"
        )
    fields = ("round", "randomness", "signature", "previous_signature")
    same = all(all(latest[x][f] == latest[RELAYS[0]][f] for f in fields) for x in RELAYS)
    lines.append(f"latest round fields identical across relays: {same}")
    rounds_to_go = r - int(latest[RELAYS[0]]["round"])
    lines.append(f"rounds from latest saved round to R = {rounds_to_go}")
    lines.append(f"chain_hash = {chain_hash}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
