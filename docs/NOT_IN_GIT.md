# Files intentionally **not** in Git

**Last updated:** 2026-10-14

This project mixes small code with very large model weights. GitHub rejects any
single file above **100 MB**, and a repo that carries hundreds of MB of weights is
slow to clone and painful to rebase. So the large/non-portable assets below are
**deliberately untracked** (listed in [`.gitignore`](../.gitignore)).

> ⚠️ **A fresh clone is NOT runnable as-is.** `checkpoint/` and most of
> `artifacts/` are empty after cloning. Follow [Bootstrap](#bootstrap-after-a-fresh-clone)
> before running any training or batch inference command.

---

## Bootstrap (after a fresh clone)

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows;  source .venv/bin/activate on *nix
pip install -r requirements.txt

# 1. Base ECGFounder weights — REQUIRED before any diagnosis-head command runs
#    Download from https://huggingface.co/PKUDigitalHealth/ECGFounder
#    and place the single-lead weights at:  checkpoint/1_lead_ECGFounder.pth

# 2. Personal autoencoder weights (~0.6 MB each).
#    Cheapest path: copy the tracked copies out of the offline bundle, or retrain:
python -m evaluation.train_segments           # writes artifacts/micah/phase1_beat{,_resting_hr}/

# 3. Diagnosis head (~117 MiB — must be retrained, it is deliberately not committed)
python -m training.ecgfounder_narrow.prepare_data   # also rebuilds processed/**/*.npz
python -m training.ecgfounder_narrow.train
python -m evaluation.calibrate_diagnosis

# 4. Rebuild the portable offline bundle (regenerates the untracked 117 MiB copy)
python -m training.build_offboard_bundle
```

Step 4 reads `checkpoint/1_lead_ECGFounder.pth` and copies it into
`offboard/portable_bundle/code/checkpoint/`. If the base checkpoint is missing,
`build_offboard_bundle` **prints a warning and silently produces a bundle whose
diagnosis heads cannot load offline** — so do step 1 first.

---

## Inventory

| Path / pattern | Size | What it is | Why excluded | How to restore |
|---|---|---|---|---|
| `.env` | — | Supabase keys / secrets | Never commit secrets | `copy .env.example .env` |
| `checkpoint/*.pth`<br/>→ `checkpoint/1_lead_ECGFounder.pth` | ~353 MiB | ECGFounder **base** single-lead weights | Above the 100 MB host limit | Download from [Hugging Face](https://huggingface.co/PKUDigitalHealth/ECGFounder) |
| `artifacts/**/*.pt`<br/>→ `artifacts/micah/phase1_beat/personal_ae.pt`,<br/>`.../phase1_beat_resting_hr/personal_ae.pt` | ~0.6 MiB each | Personal autoencoder weights + `bundle.json` thresholds | Regenerated locally (see the `.gitignore` comment) | `python -m evaluation.train_segments`<br/>*or* copy from the tracked bundle at `offboard/portable_bundle/artifacts/phase1_beat*/` |
| `artifacts/**/*.pt`<br/>→ `artifacts/micah/diagnosis_narrow/diagnosis_narrow.pt`,<br/>`artifacts/micah/diagnosis_narrow_sagemaker/diagnosis_narrow.pt` | ~117 MiB each | ECGFounder **narrow diagnosis head** (local + SageMaker) | Above the host limit | `python -m training.ecgfounder_narrow.train` + `python -m evaluation.calibrate_diagnosis` |
| `offboard/portable_bundle/artifacts/diagnosis_narrow/*.pt` | ~117 MiB | Bundle's copy of the diagnosis head | Same 100 MB limit | `python -m training.build_offboard_bundle` |
| `processed/**/*.npz` | — | Preprocessed train/val shards | Regenerable, large | `python -m training.ecgfounder_narrow.prepare_data` |
| `offboard/aws_mirror/` | — | AWS offboard mirrors | AWS account is closed; keep local only | N/A (defunct) |

### Gitignore anchoring subtlety

Patterns that contain a `/` (like `artifacts/**/*.pt`) are **anchored to the repo
root**, so `artifacts/**/*.pt` does **not** match
`offboard/portable_bundle/artifacts/...`. That is why the bundle copy needs its own
explicit rule. If you add a new nested `artifacts/` directory, add a matching rule.

Note also that `offboard/portable_bundle/artifacts/**/personal_ae.pt` (~0.6 MiB) is
**small and IS tracked on purpose** — the bundle is meant to be self-contained for
the personal-AE half. Only the ~117 MiB diagnosis head inside the bundle is ignored.

---

## Large files that *are* tracked

These are under the 100 MB limit so they were committed, but they are still sizable
and inflate the pack:

| File | Size |
|---|---|
| `reports/batch_micah_full/SBX-1783615111746_scores.jsonl` | ~69 MB |
| `reports/batch_micah_full/SBX-1783926162025_scores.jsonl` | ~53 MB |
| `reports/batch_micah_full/SBX-1783526750041_scores.jsonl` | ~24 MB |

---

## Discrepancy: `exports/` is tracked despite the policy

`README.md` and `docs/AGENT_CONTEXT.md` §11 state that raw Sandbox ECG must
**never** be committed, and `exports/` is annotated `[gitignored — do not commit]`.

**In reality `exports/` is tracked and has no `.gitignore` rule:** ~18,130 files,
~144 MB, under `exports/2026-07-16T162705Z/`. No single file exceeds 100 MB, so it
does not block a push — but it does mean raw physiological data lands on the remote
(the repo is currently **public**), and the documented policy is currently false.

Two options, not yet decided:

- **Untrack it** (`git rm -r --cached exports` + add `exports/` to `.gitignore`) —
  matches the documented policy; existing copies stay on disk but are no longer
  shared. Note this is a large removal and any clone would then need the raw export
  delivered out-of-band.
- **Keep it tracked** — then correct `README.md` / §11 so the docs stop claiming it
  is ignored.

---

## Adding new large files

- Anything over ~50 MB: think twice. Over 100 MB: the push will be **rejected**.
- Options: regenerate locally, publish as a GitHub Release asset, store out-of-band
  (S3/Drive), or adopt **Git LFS** (GitHub free tier = 1 GB storage + 1 GB/month
  bandwidth; note every clone spends that bandwidth, and collaborators need
  `git-lfs` installed or they silently receive pointer files).
- Whatever you choose, add a row to the inventory above and a rule + comment to
  `.gitignore` so the next person is not left guessing.
