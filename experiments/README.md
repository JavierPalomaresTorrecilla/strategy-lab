# experiments/

Every research experiment (a hypothesis tested against data with a specific
configuration) should be recorded in `registry.csv` for provenance and audit.

## Why

Recording experiments prevents silent parameter mining: if only "winning" runs
are remembered, the apparent success rate of a research process is meaningless.
Recording every run, including failures, is what makes results falsifiable.

## What to record

At minimum, per experiment: date, hypothesis, strategy/config identifier,
data range (train/validation/test), and outcome/notes. See `registry.csv` header.
