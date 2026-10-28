# OTS upgrade and archive-path pretest (2026-10-28)

Stephen ticked both as below-floor jobs: "U1 Upgrade OTS proofs" and "U2 Archive-path pretest".

## U1: OTS proof upgrade

- **Where it ran.** On instance 2 (209.137.198.191; host key TOFU/UNVERIFIED), in
  /home/Ubuntu/oac-v3-recheck-20261028T030002Z-26737a55/ots-upgrade. The tooling was a venv
  (miniconda CPython 3.12.4) with opentimestamps 0.4.5 and `scripts/v3_ots_upgrade.py`.
- **Calendars.** All 4 calendars served upgrades for both proofs: alice/bob.btc.calendar.opentimestamps.org,
  btc.calendar.catallaxy.com and finney.calendar.eternitywall.com.
- **Bitcoin attestations** (MEASURED): heights 972699, 972700, 972728 and 972749, the same for
  both proofs. On both explorers (blockstream.info and mempool.space), the explorers agree and the
  attested merkle roots match.
- **T_fwd** (prereg §3.5 forward median of H..H+10, MEASURED):
  - 972699: 2026-10-23T03:30:22Z, the earliest
  - 972700: 03:30:49Z
  - 972728: 07:27:50Z
  - 972749: 10:15:26Z
- **Independent re-check from the laptop.** I recomputed the earliest calendar's T_fwd (H=972699)
  from both explorers: merkle root ef2f99a0…, T_fwd 2026-10-23T03:30:22Z. That matches.
- **Conditions.**
  - Registration (digest 8d3f31ca…): T_fwd 2026-10-23T03:30:22Z is earlier than
    2026-10-24T00:00:00Z (§13 step 1 and the r1 disclosure). It is the registration's dated
    time; the registration has been public since its tag push on 2026-10-14.
  - r1 (digest fae741f6…): T_fwd 2026-10-23T03:30:22Z is earlier than 2026-10-24T00:00:00Z and
    at least 2 h before R′ (2026-11-21T00:00:00Z).
- **Files.**
  - The upgraded proofs replace v3/timestamps/*.ots.
  - The pending originals are kept as logs/v3/timestamps/*.pending_20261023.ots.
  - The full report is upgrade_report_20261028.json.
- **Still to do.** A formal check by the §3.3 seal checker (`v3b-seal`) is still pending. The
  numbers above come from this script plus a manual explorer re-check.

## U2: archive-path pretest

- **Requests** (MEASURED, 2026-10-28T03:35:25Z):
  - **Software Heritage** "save code now" for https://github.com/szl-holdings/szl-forge: accepted,
    id 2585085, task pending.
  - **Internet Archive**, oac-v3-registered: the tag ref was captured as web/20261028033539, about
    14 s after the request.
  - **Internet Archive**, oac-v3-registered-r1: captured as web/20261028033545.
- **Still to do.** The Software Heritage visit date must still be read back; it gives the
  archive delay for T2 planning.
