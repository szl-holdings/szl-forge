# Payload contract (second-reader) and predicate-AND

Owner: `szl-holdings/szl-forge`.
Companion collector (do not replace): `tools/observe_public_estate_files.py`.
Companion release authorizer (do not replace): `tools/authorize_invariants_release.py`.

`szl_payload_contract.py` is an offline second-reader for digest-bound census ZIPs
and fixed-k consistency matrices. It is not a collector, publisher, runtime probe,
or authorizer.

`szl_predicate_and.py` is an offline dream-gate. Independent predicates AND.
Unknown is HOLD. Incomparable populations cannot be summed. ALTK metrics cannot
collapse. Forbidden promotion language is classified, not executed. This module
cannot emit `production_authorization=true`.

```
python -m unittest discover -s tools -p test_szl_payload_contract.py -v
python -O -m unittest discover -s tools -p test_szl_payload_contract.py -v
python -m unittest discover -s tools -p test_szl_predicate_and.py -v
python -O -m unittest discover -s tools -p test_szl_predicate_and.py -v
```

Passing tests do not qualify the estate. `production_authorization` stays false.
LFS identities may be REDACTED; pointer hashes are not content digests.
RUNNING is not runtime-verified.
HTTP 200 is OBSERVED, never LIVE.
AGI / ALL_DONE / all-green are FORBIDDEN_PROMOTION.
