# Phase 1 Decision Record

**Date:** 2026-07-17  
**Status:** Personal baseline **PASS** — proceed with beat-aligned loop

## Investigation summary

We compared strategies against wearable ECG literature (personal AE, beat-level morphology, subject calibration) and our Phase 0 results.

| Approach | False alert @ P99 | Result |
|----------|-------------------|--------|
| 10s windows + temporal recording split | 6.7–7.0% | Fail |
| 10s windows + stricter QC + larger AE | 7.0% | Fail |
| 10s windows + random recording split | N/A | Eval set had no rest windows |
| **1s beat-aligned + window-level split + val threshold** | **1.75%** | **Pass** |

## Root causes of 10s window failure

1. **Recording-level temporal split** — later recordings drift from earlier baseline (weeks of wear).
2. **Recording-level random split** — eval recordings lacked sedentary-tagged beats under strict QC.
3. **10s arbitrary windows** — mix multiple morphologies; literature uses **1s beat-aligned** segments for wearable anomaly detection.

## Winning architecture (v1)

```
Sandbox SBEC → beat parse → 1s R-peak window @ 500 Hz
    → QC (sedentary, no motion artifact)
    → personal conv AE
    → reconstruction error vs validation threshold
    → alert on sustained deviation (future: persistence layer)
```

## Explicitly deferred

| Item | Why |
|------|-----|
| ECGFounder zero-shot diagnosis gate | 99.5% false suspicious on Polar data |
| ECGFounder fine-tune | After personal loop validated; needs SageMaker GPU |
| AWS S3 upload | When multi-user or off-machine archive needed |
| SageMaker | Phase 2–3 when local training doesn't scale |
| On-device inference | Phase 5 |

## Next implementation steps

1. ~~Add **persistence / smoothing** on reconstruction error~~ — `adaptation/persistence.py` + batch pipeline.
2. ~~Implement **fusion gating** skeleton~~ — `fusion/gating.py`.
3. **Segment by activity** (rest vs walk) as separate fingerprints.
4. Plan **ECGFounder fine-tune** on narrow rhythm labels only (not 150-class zero-shot).
5. ~~S3 migration~~ — `python -m datasets.s3_sync`; SageMaker jobs next.

## Run command

```bash
python -m evaluation.personal_baseline_phase1 --config configs/phase1_beat_micah.yaml
```

Report: `reports/phase1_beat/phase1_beat_report.md`
