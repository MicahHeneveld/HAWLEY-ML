# HAWLEY ML — offline bundle

Built before AWS account closure.

- `artifacts/` — personal AE (rest, resting_hr) + ECGFounder narrow diagnosis head
- `code/` — batch scoring + fusion (set PYTHONPATH=code)
- `configs/` — batch_micah_full.yaml
- `vendor/ECGFounder/` under code/ for diagnosis load

```bash
pip install -r requirements.txt
set PYTHONPATH=code
python -m inference.batch_pipeline --config configs/batch_micah_full.yaml
```

No AWS required.
