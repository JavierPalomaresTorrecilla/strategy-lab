# data/

**These directories are placeholders/documentation only.** They exist to
establish the raw/processed distinction conceptually and to hold small,
committed synthetic fixtures if that's ever needed — they are not where real
datasets live, starting with Milestone 2.

- `raw/` — conceptually: unmodified source data exactly as retrieved (never
  edited in place).
- `processed/` — conceptually: canonical/derived datasets produced
  deterministically from `raw/` by code in `src/strategy_lab/data/`.

All real snapshot data — raw provider downloads and their canonicalized
derivatives — is stored exclusively under the path named by the
`STRATEGY_LAB_DATA_ROOT` environment variable, **outside this Git repository**
and outside these `data/raw`/`data/processed` directories. See
`strategy_lab.data.env`, `strategy_lab.data.snapshot`, and
`strategy_lab.data.processed` for the resolution, path layout, and
metadata/checksum scheme. If `STRATEGY_LAB_DATA_ROOT` is unset or
unavailable, data-writing code fails closed — it never falls back to writing
large data into this directory, or anywhere else inside the repo, or onto
the internal disk.

Datasets should be reproducible from documented sources. If a dataset cannot be
regenerated from raw data plus code, treat that as a gap to fix, not a convenience.
