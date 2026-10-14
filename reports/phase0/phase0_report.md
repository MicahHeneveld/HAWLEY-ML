# Phase 0 Feasibility Report

Generated: 2026-07-17T08:40:02.141335+00:00

## Dataset
- Export: `exports\2026-07-16T162705Z`
- Recordings: 31 (21 train / 10 eval)
- Clean rest windows: 3000 train / 1000 eval

## Personal Baseline (Option A)
- Eval false alert rate @ P99: **6.70%
- Alert threshold (train reconstruction MSE): 1.011159
- Eval error mean / p95: 0.973614 / 1.019404

## Diagnosis Gate (Option B — ECGFounder)
- Status: ran
- Suspicious rate (pathology prob >= 0.5): **99.50%**
- Pathology strength mean / p95: 0.7901 / 0.9399
- Normal strength mean: 4.6196
- Top predicted labels:
  - ABNORMAL ECG: 650
  - SINUS RHYTHM: 336
  - NORMAL SINUS RHYTHM: 14

## Go / No-Go
- Personal baseline pass: **False**
- ECGFounder pass: **False**
- Overall Phase 0 pass: **False**

### Notes
- Personal: Target eval false alert rate < 5% at 99th percentile threshold.
- ECGFounder: Diagnosis gate interim proxy suspicious_rate should be < 10% on clean rest.
