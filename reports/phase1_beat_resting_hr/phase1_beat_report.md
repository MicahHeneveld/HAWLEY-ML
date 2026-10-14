# Phase 1 Beat-Aligned Personal Baseline

Generated: 2026-07-17T10:55:27.111327+00:00

## Approach
- **Window mode:** beat (1 s @ 500 Hz, R-peak centered)
- **Split:** window_random
- **Beats:** 5000 train / 2000 eval

## Personal Baseline
- Eval false alert @ P99: **0.70%**
- Alert threshold: 0.998378
- Eval error mean / p95: 0.511650 / 0.985099
- Eval sustained false alert (smoothed + consecutive): **0.00%**

## Go / No-Go
- Personal baseline pass: **True**
- Overall Phase 1 pass: **True**

## Notes
- Beat-aligned 1s windows + window-level split (single-subject personalization).
- Deferred — zero-shot failed; fine-tune only after personal baseline passes.
- S3 sync via python -m datasets.s3_sync; SageMaker deferred to Phase 3.
- Sustained alert: 5-beat MA + 3 consecutive beats.
