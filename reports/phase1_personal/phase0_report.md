# Phase 0 Feasibility Report

Generated: 2026-07-17T08:55:08.180212+00:00

## Dataset
- Export: `exports\2026-07-16T162705Z`
- Recordings: 31 (21 train / 10 eval)
- Clean rest windows: 3000 train / 1000 eval

## Personal Baseline (Option A)
- Eval false alert rate @ P99: **7.00%
- Alert threshold (train reconstruction MSE): 1.007403
- Eval error mean / p95: 0.967872 / 1.016035

## Diagnosis Gate (Option B — ECGFounder)
- Status: skipped
- Reason: config or --skip-ecgfounder

## Go / No-Go
- Personal baseline pass: **False**
- ECGFounder pass: **None**
- Overall Phase 0 pass: **False**

### Notes
- Personal: Target eval false alert rate < 5% at 99th percentile threshold.
- ECGFounder: ECGFounder not run — download checkpoint to complete diagnosis gate.
