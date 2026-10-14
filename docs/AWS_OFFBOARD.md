# Offline operation (post-AWS)

**The AWS account is closed** (sandbox decommissioned at the Organizations level — console, S3, SageMaker, and CLI all return explicit-deny). The project runs entirely local now. AWS is no longer part of the plan.

## What was completed on AWS (historical)

- Raw Sandbox export synced to `s3://hawley-ml-ecg/raw/micah/`.
- One-time **SageMaker CPU training** of the ECGFounder narrow head (job `hawley-dx-1784301182`) — output fetched locally to `artifacts/micah/diagnosis_narrow_sagemaker/` (calibrated threshold 0.75).
- Artifacts / reports / processed shards backed up to S3.

All of that data was produced from local sources and is still on disk, so **nothing is stranded in the cloud**. The S3 backups are inaccessible but redundant.

## Everything needed lives locally

| Asset | Location |
|-------|----------|
| Personal AE (rest, resting_hr) | `artifacts/micah/phase1_beat`, `artifacts/micah/phase1_beat_resting_hr` |
| Local diagnosis head | `artifacts/micah/diagnosis_narrow` |
| SageMaker-trained diagnosis head | `artifacts/micah/diagnosis_narrow_sagemaker` |
| Base ECGFounder weights | `checkpoint/1_lead_ECGFounder.pth` |
| Raw ECG | `exports/2026-07-16T162705Z/` |
| Portable, self-contained bundle | `offboard/portable_bundle/` (+ `hawley_portable_bundle.zip`) |

## Run everything locally

```bash
# Train / retrain personal segment models
python -m evaluation.train_segments

# Calibrate the diagnosis threshold (local or SageMaker artifact)
python -m evaluation.calibrate_diagnosis
python -m evaluation.calibrate_diagnosis --artifact-dir artifacts/micah/diagnosis_narrow_sagemaker

# Full batch score
python -m inference.batch_pipeline --config configs/batch_micah_full.yaml
```

## Rebuild the portable offline bundle

```bash
python -m training.build_offboard_bundle
```

This copies all model artifacts (`.pt` weights + `bundle.json`), the base ECGFounder checkpoint (into `code/checkpoint/`), inference code, vendored ECGFounder, configs, and reports into `offboard/portable_bundle/`, then writes `bundle_manifest.json` with a `weight_files` list to verify completeness. Zip it:

```bash
# Windows PowerShell
Compress-Archive -Path offboard\portable_bundle\* -DestinationPath offboard\hawley_portable_bundle.zip -Force
```

The bundle runs with no cloud:

```bash
cd offboard/portable_bundle
pip install -r requirements.txt
set PYTHONPATH=code
python -m inference.batch_pipeline --config configs/batch_micah_full.yaml
```

## AWS scripts removed (2026-07-18)

The cloud-only scripts were deleted since the account is closed:

- `datasets/s3_sync.py`, `datasets/sync_to_s3.py`, `datasets/s3_io.py`
- `training/aws_offboard.py`
- `training/sagemaker/` (launch, train, fetch, role, processing)

`training/prepare_offboard.py` was kept but stripped to a **local-only** wrapper (calibrate + build bundle). `boto3` is now optional in `requirements.txt`.
