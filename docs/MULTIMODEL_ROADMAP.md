# Multi-model ECG — decision log

## Investigation (2026-07-17): “5 recordings” vs 31

Not a loader bug. See `reports/coverage/dataset_coverage.md`:

- **25** recordings have chunk files; **5** have beats that pass Layer C QC **and** route to `rest` / `resting_hr` models.
- **20** long sessions tag **`stationary_activity`** but almost all beats also carry **`motion_artifact`**, so Layer C QC correctly vetoes them.
- Expanding `sedentary_stationary` → include `stationary_activity` does **not** increase routable beats until motion tags improve at capture or QC rules change for a separate “motion-tolerant” research track.

## Best next step (literature + project state)

Wearable pipelines ([PLOS Digital Health federated AE](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000793), [Computers in Biology and Medicine 2024 VAE](https://doi.org/10.1016/j.compbiomed.2024.109422)) use:

1. **Personal / unsupervised reconstruction** on clean normal beats — **done** (rest + resting_hr).
2. **Explicit fusion + QC veto** before adding a population classifier — **implemented** (`fusion/pipeline.py`, batch JSONL fields).
3. **Second model** (ECGFounder fine-tuned head, not zero-shot) — **done** (narrow head trained locally and once on SageMaker; both bundles local).

The ECGFounder narrow head is now the fusion `diagnosis_suspicious` signal. It is **still blocked on real labels** — current labels are weak Sandbox-tag proxies, not clinical pathology.

## Implemented this pass

- `preprocessing/tag_schema.py` — tag aliases, QC veto, segment routing
- `fusion/pipeline.py` — multi-model orchestration slot
- `evaluation/dataset_coverage.py` — corpus truth table
- Batch scores include `qc_pass`, `qc_veto_reason`, `model_slot`

## ECGFounder narrow head (implemented)

Weak-label 2-class linear probe on 1-lead ECGFounder (10 s @ 500 Hz):

```bash
# Requires HF weights in checkpoint/. Runs locally (CPU/GPU) — no cloud.
python -m training.ecgfounder_narrow.prepare_data
python -m training.ecgfounder_narrow.train
```

Batch fusion uses `diagnosis_suspicious` from `artifacts/micah/diagnosis_narrow_sagemaker/` (or `diagnosis_narrow/`).

## Next implementation queue (all local — AWS is out of scope, account closed)

1. ~~ECGFounder narrow-head fine-tune~~ + fusion wiring — **done**
2. ~~Download trained head from cloud before account closure~~ — **done** (`artifacts/micah/diagnosis_narrow_sagemaker/`)
3. **Replace weak tag labels** with clinician- or device-validated labels, then retrain the narrow head locally
4. Finish the full diagnosis-integrated batch run locally (last run was interrupted)
5. Phase 4: fusion tuning + LLM narrator on structured JSON
