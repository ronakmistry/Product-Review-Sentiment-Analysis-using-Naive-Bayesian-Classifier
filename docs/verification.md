# Restoration verification

Validation date: 2026-10-21. Platform: macOS Apple Silicon, CPython 3.12.14.
Dependencies: `requirements-lock.txt`; `pip check` reported no broken requirements.

## Completed

- `4` tests passed with `python -m pytest -q`.
- All 1 supported root notebook(s) executed from fresh Jupyter kernels.
- Executed copies and HTML previews are under ignored `artifacts/notebooks/`.
- HTML checked for expected headings, fixture labeling, and successful outputs.
- Original notebooks preserved byte-for-byte under `historical/`; starting commit is in `inventory.json`.
- Repository-relative README links checked against existing files.
- Training and save/load verification used generated fixtures only.

## Limits

No original datasets, pretrained weights, or recovered laptop resources were used.
No real-data performance was reproduced. Historical scores remain unverified.
The dependency environment was created cleanly for this restoration and shared across the
four projects; per-project locks record their dependency closures. Linux CI is configured,
but its result must be checked separately on the PR.

Generated plots were visually inspected: labels and layout were legible.
Browser-based full-page HTML layout review was unavailable because local-file browser access was blocked.
