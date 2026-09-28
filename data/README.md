# data/

Local data storage for research. Not committed to version control (see `.gitignore`).

- `raw/` — unmodified source data exactly as retrieved (never edited in place).
- `processed/` — cleaned/derived datasets produced deterministically from `raw/` by code in `src/strategy_lab/data/`.

Datasets should be reproducible from documented sources. If a dataset cannot be
regenerated from raw data plus code, treat that as a gap to fix, not a convenience.
