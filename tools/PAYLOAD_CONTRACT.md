# Payload contract (second-reader)

Owner: `szl-holdings/szl-forge`.
Companion collector (do not replace): `tools/observe_public_estate_files.py`.

`szl_payload_contract.py` is an offline second-reader for digest-bound census ZIPs
and fixed-k consistency matrices. It is not a collector, publisher, runtime probe,
or authorizer.

```
python -m unittest discover -s tools -p test_szl_payload_contract.py -v
python -O -m unittest discover -s tools -p test_szl_payload_contract.py -v
```

Passing tests do not qualify the estate. `production_authorization` stays false.
LFS identities may be REDACTED; pointer hashes are not content digests.
RUNNING is not runtime-verified.
