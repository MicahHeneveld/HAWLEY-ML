# HAWLEY ML — Agent Handoff Document

**Last updated:** 2026-10-14  
**Source project:** SmashBeat (`C:\SmashBeat\smashbeat`)  
**Target project:** HAWLEY ML (this repo — separate from SmashBeat Flutter app)  
**Status:** Phase 1 personal baseline **PASS** (1.75% false alert) — beat-aligned 1s windows; ECGFounder narrow head trained (local + one SageMaker run); running fully offline

This document captures everything an agent or developer needs to continue the ECG ML project without re-discovering context from SmashBeat or prior brainstorming sessions.

> **⚠️ AWS is out of scope (2026-07-18).** The sandbox AWS account was closed. All cloud work (one SageMaker training job, S3 backups) is complete and its outputs are local. The project now runs **entirely offline** — ignore SageMaker/S3 references below except as historical record. Training source of truth is the local `exports/` archive; models live in `artifacts/micah/`. See [`docs/AWS_OFFBOARD.md`](AWS_OFFBOARD.md) for offline operation.

---

## 1. Mission (One Sentence)

Build a **personalized, multi-model ECG analysis system** that learns from longitudinal Polar H10 data captured via SmashBeat Sandbox, adapts to individual users over time, and eventually supports daily rhythm/health insights — starting with one subject (Micah) before scaling to a future dedicated capture app.

---

## 2. Scope Boundaries

### In scope (v1)
- Use **existing SmashBeat Sandbox captures only** as longitudinal training data
- First corpus: **Micah's subject folder** in Supabase Storage (not a separate bucket)
- Offline/batch ML pipeline in **HAWLEY ML** (Python, local CPU/GPU; ~~SageMaker, S3~~ — AWS closed)
- Phase 0 feasibility: ECGFounder + personal baseline on real Polar data
- Multi-model fusion architecture (design + prototype)
- Closed-loop adaptation concept (predictive coding / personal fingerprint, not full robotics RL)

### Out of scope (v1)
- **No changes required to SmashBeat** for Phase 0–2 (capture already works)
- **No new capture app** until ML loop is validated on Micah data
- **No on-device inference** until models are validated in batch
- **No clinical diagnosis claims** — wellness/research framing until clinician validation
- **No LLM-as-primary-classifier** — LLM is narrator on structured JSON only

### SmashBeat vs HAWLEY ML roles

| Repo | Role |
|------|------|
| **SmashBeat** | Flutter research app. Polar H10 capture, Sandbox recording, Supabase upload. Stays as-is. |
| **HAWLEY ML** | Dataset loaders, preprocessing, local training, evaluation, adaptation, fusion, model artifacts. |

---

## 3. Strategic Decisions Already Made

Do not re-litigate these without explicit user approval:

1. **Base model:** [ECGFounder](https://github.com/PKUDigitalHealth/ECGFounder) (NEJM AI 2025, Hugging Face weights) for deep analysis
2. **Training platform:** ~~AWS SageMaker~~ → **local (CPU/GPU)**. AWS account is closed; the one SageMaker job that ran is done and its output is local. All future training is local.
3. **Data source of truth (training):** local **`exports/`** archive (originally exported from Supabase; the S3 mirror is defunct). Supabase remains capture buffer only.
4. **Personalization approach:** Tier 1 personal baseline fingerprint first; Tier 2 per-user adapters later; defer on-device continual learning
5. **Dual-goal adaptation loop:**
   - **Primary goal (Option A):** Minimize error vs personal cardiac baseline (autoencoder / fingerprint)
   - **Secondary goal (Option B):** Diagnosis model should classify "normal" most of the time; high-confidence pathology on clean signal = alert
6. **Multi-model fusion:** Explicit DAG with QC veto, context conditioning, agreement gating — not naive ensemble
7. **Robotics analogy:** Use **predictive coding / observe-predict-error-update** loop, not action-based RL (no actuators on the heart)

---

## 4. Upstream Data: SmashBeat Sandbox

### What Sandbox is
Android-only longitudinal ECG capture mode. Separate from SmashBeat **Live** sessions (which strip raw ECG before Supabase upload). Sandbox runs a native foreground service, chunks ECG every 60 seconds, and uploads to Supabase.

**Live and Sandbox cannot run simultaneously** on one Polar H10 connection.

### Storage layout (Supabase)
- **Bucket name:** `sandbox-ecg` (single bucket for all subjects)
- **Micah's data:** `{subject_id}/{recording_id}/chunk_{index}.bin.zst` — confirmed `subject_id = "Micah"`; there is no separate "Micah bucket"
- **Local archive (HAWLEY ML):** `exports/2026-07-16T162705Z/` — 31 complete recordings, ~9,039 chunks (~150 h). Incomplete `SBX-1783694914924` removed.
- **Postgres tables:** `sandbox_recordings`, `sandbox_chunks` (see `smashbeat/sandbox-migration.sql`)
- **RLS:** Permissive open policies — research/prototype only, not production-ready

### Binary ECG format (`chunk_XXXXX.bin` — uncompressed; stored as `.bin.zst`)

| Offset | Size | Field |
|--------|------|-------|
| 0 | 4 | Magic `SBEC` (bytes) |
| 4 | 4 | Format version `1` (int32 LE) |
| 8 | 4 | Sample rate `130` (int32 LE) |
| 12 | 4 | Sample count N (int32 LE) |
| 16 | 4×N | Samples as int32 LE **microvolts** |

- **Sample rate:** 130 Hz (Polar H10 standard everywhere in codebase)
- **Chunk duration:** 60 seconds → ~7,800 samples/chunk
- **Checksum in Postgres:** SHA-256 of **uncompressed** `.bin` (zstd object hash differs)

### Beat metadata (`chunk_XXXXX_beats.jsonl.zst`)
One JSON object per R-peak (from native `SandboxBeatProcessor.kt` — simpler than Dart `EcgBeatSegmenter`):

```json
{
  "beat_index": 42,
  "sample_offset": 3150,
  "elapsed_ms": 24231,
  "rr_ms": 812,
  "hr_bpm": 74,
  "rmssd_ms": 28.3,
  "tags": ["sedentary_stationary", "resting_hr", "clean_signal"],
  "features": {
    "acc_rms_mg": 35.2,
    "gps_speed_mps": null,
    "phone_accel_rms": 0.8,
    "activity_label": "STILL",
    "transport_confirmed": false,
    "peak_confidence": 0.72,
    "polar_contact_ok": true,
    "signal_quality": "good"
  }
}
```

**Tag schema v2:** `smashbeat/lib/src/sandbox/tags_schema.json`

| Layer | Purpose | Example tags |
|-------|---------|--------------|
| A | Kinematic context | `sedentary_stationary`, `walking`, `phone_motion` |
| B | Physiological context | `resting_hr`, `low_hrv`, `cognitive_stress` |
| C | Signal quality | `clean_signal`, `motion_artifact`, `poor_contact` |

**Treat all tags as weak labels, not ground truth.**

**Implementation note:** Beat `tags` in exported JSONL are often a **string** like `"[sedentary_stationary, phone_motion]"`, not a JSON array. HAWLEY ML `preprocessing/qc.py` parses both formats.

### Export tool (use this first)
```bash
cd smashbeat/tools
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # SUPABASE_URL + SUPABASE_ANON_KEY
python export_sandbox.py        # exports to smashbeat/exports/<timestamp>/
```

Export layout:
```
exports/<timestamp>/
  SBX-<id>/
    recording.json
    chunks.csv
    export_summary.json
    chunks/
      chunk_00000.bin.zst
      chunk_00000_beats.jsonl.zst
  batch_summary.json
```

**Missing dependency:** Add `zstandard` to HAWLEY ML for decompression (not in SmashBeat `tools/requirements.txt` today).

**Useful flags:** `--recording-id`, `--completed-only`, `--purge` (only after verified export)

See `smashbeat/tools/README.md` for full workflow.

---

## 5. Domain Gap: Polar H10 vs ECGFounder

| | Polar / Sandbox | ECGFounder |
|--|-----------------|------------|
| Sample rate | 130 Hz | 500 Hz |
| Window | 60s chunks | 10s windows (5000 samples) |
| Leads | Single chest lead | 12-lead clinical; single-lead variant exists |
| Preprocessing | Raw µV | Bandpass 0.5–50 Hz, notch, **z-score per lead** (strict — see ECGFounder `dataset.py`) |
| Labels | Rule-based tags | 150 cardiologist-derived diagnostic categories |
| Motion | Common (ACC tags available) | Hospital/clinical setting |

**Phase 0 go/no-go:** Can ECGFounder + preprocessing produce acceptable rhythm/normal-vs-abnormal behavior on Micah's **clean rest** windows without excessive false alerts?

**Preprocessing bridge (must build):**
1. Decompress zstd
2. Parse SBEC binary
3. Align beats JSONL by `sample_offset`
4. Resample 130 Hz → 500 Hz
5. Extract sliding 10s windows
6. Apply ECGFounder filter + normalization exactly
7. QC-gate using Layer C tags and/or signal metrics

---

## 6. Target Architecture (HAWLEY ML)

```
SmashBeat Sandbox → Supabase (capture buffer)
        ↓ export (one-time)
Local raw archive: exports/ + manifest (source of truth for training)
        ↓ preprocessing (local)
Preprocessed windows + QC flags + subject splits
        ↓ training (local CPU/GPU)
ECGFounder narrow head / personal autoencoder → artifacts/micah/
        ↓
Batch inference + threshold calibration (local)
        ↓ (future)
On-device lightweight model + fusion in separate app
```

*(Original design routed this through S3 + SageMaker; AWS is closed, so every stage now runs locally.)*

### Multi-model DAG (fusion — implement explicitly)

```
Polar window
  → Layer 0: Signal QC (motion, contact, noise) — VETO power
  → Layer 1: Personal baseline error (autoencoder / fingerprint)
  → Layer 2: ECGFounder diagnosis head (150-class or fine-tuned subset)
  → Fusion: agreement gating
  → Layer 3: LLM narrator (structured JSON in, text out)
```

**Alert tiers:**
| State | Condition |
|-------|-----------|
| Stable | Personal error low + normal diagnosis labels dominate |
| Personal anomaly | Personal error high, diagnosis still normal |
| Suspicious | Diagnosis fires, personal error low — re-sample / stillness prompt |
| Critical | Both agree abnormal on clean signal |
| Reject | QC failed — no alert, no model update |

### Adaptation timescales

| Timescale | Mechanism | Where |
|-----------|-----------|-------|
| Per-window | QC + context segmentation | Batch / future edge |
| Daily | Threshold adaptation from error distribution | Local → config |
| Weekly | Adapter / autoencoder retrain on new export | Local (CPU/GPU) |

**Do not implement online RL weight updates in v1.**

---

## 7. Recommended HAWLEY ML Repo Structure

```
hawley-ml/
├── docs/
│   └── AGENT_CONTEXT.md          # pointer to this handoff + project-specific updates
├── datasets/
│   ├── sbec_parser.py            # SBEC binary reader
│   ├── supabase_export.py        # port/extend export_sandbox.py
│   ├── manifest.py               # dataset manifest schema
│   └── s3_sync.py
├── preprocessing/
│   ├── resample.py               # 130 → 500 Hz
│   ├── ecgfounder_prep.py        # filter, notch, z-score (match dataset.py exactly)
│   └── qc_filter.py              # Layer C + signal metrics
├── models/
│   ├── ecgfounder_wrapper.py
│   ├── personal_autoencoder.py
│   └── rhythm_cnn.py             # future lightweight model
├── adaptation/
│   ├── fingerprint.py            # personal baseline
│   ├── thresholds.py             # percentile-based alert gates
│   └── lora_user_adapter.py      # Phase 3
├── fusion/
│   └── gating.py                 # explicit fusion rules
├── training/
│   └── sagemaker/                # job definitions, Docker if needed
├── evaluation/
│   └── phase0_feasibility.py     # FIRST CODE TO WRITE
├── inference/
│   └── batch_pipeline.py
├── notebooks/
│   └── phase0_micah_eda.ipynb
├── configs/
│   └── subjects/micah.yaml
├── requirements.txt
└── README.md
```

---

## 8. Phased Roadmap

## Phase 0 — Feasibility (complete)
## Phase 1 — Personal baseline (PASS)

See [`docs/PHASE1_DECISION.md`](PHASE1_DECISION.md) for investigation rationale.

**Winning approach:** 1s beat-aligned windows, window-level 80/20 split, validation-set P99 threshold.

```bash
python -m evaluation.personal_baseline_phase1 --config configs/phase1_beat_micah.yaml
```

Report: `reports/phase1_beat/phase1_beat_report.md` — **1.75% false alert** on Micah clean rest beats.

### Phase 1 — Data platform (when needed)
- S3 bucket + manifest format
- Automated Supabase → S3 export
- Subject-level train/val splits (never leak same person across splits)

### Phase 2 — Personal fingerprint (Option A)
- Segment by activity tag (rest vs walk vs exertion)
- Per-segment autoencoder for Micah
- Nightly error scoring + threshold calibration

### Phase 3 — Diagnosis gate (Option B)
- Fine-tune ECGFounder on narrow rhythm labels (not all 150)
- Fusion with personal baseline
- Calibrate alert tiers

### Phase 4 — Multi-model + LLM
- Structured JSON output schema
- LLM daily narrative (wellness framing)
- Optional user feedback hook ("felt fine" / "felt off")

### Phase 5 — Future (out of HAWLEY ML v1)
- New capture/inference app
- On-device ONNX/LiteRT
- Multi-user scale, consent, proper security

---

## 9. Key SmashBeat Files (Reference Only — Do Not Fork Flutter)

| Purpose | Path |
|---------|------|
| Sandbox Postgres schema | `smashbeat/sandbox-migration.sql` |
| Export script | `smashbeat/tools/export_sandbox.py` |
| Export docs | `smashbeat/tools/README.md` |
| Tag schema | `smashbeat/lib/src/sandbox/tags_schema.json` |
| R-peak (Dart, more robust) | `smashbeat/lib/sandbox/ecg_beat_segmenter.dart` |
| Native beat processor | `smashbeat/android/.../sandbox/SandboxBeatProcessor.kt` |
| Chunk binary writer | `smashbeat/android/.../sandbox/SandboxChunkWriter.kt` |
| Sandbox upload | `smashbeat/android/.../sandbox/SandboxChunkUploader.kt` |
| Sample rate constant | `polarH10EcgSampleRateHz = 130` in `ecg_beat_segmenter.dart` |

**Recommendation:** Re-run R-peak detection offline using Dart algorithm logic (port to Python) rather than trusting native Sandbox peaks for beat-aligned training.

---

## 10. External Resources

| Resource | URL | Notes |
|----------|-----|-------|
| ECGFounder repo | https://github.com/PKUDigitalHealth/ECGFounder | Strict preprocessing in `dataset.py` |
| ECGFounder weights | https://huggingface.co/PKUDigitalHealth/ECGFounder | Place in `./checkpoint` per README |
| ECGFounder paper | https://arxiv.org/abs/2410.04133 | Input: 12×5000 or single-lead variant |
| pECG + adaptive ML review | https://pmc.ncbi.nlm.nih.gov/articles/PMC10843583/ | Distributed Scale-I/II/III architecture |
| UAD filter for wearables | https://arxiv.org/html/2510.26501 | Deep SVDD upstream of classifier |
| SageMaker LoRA adapters | https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-adapt.html | Per-user adapter pattern |
| Polar H10 SDK docs | https://github.com/polarofficial/polar-ble-sdk | 130 Hz ECG in µV |

---

## 11. Environment & Secrets

### SmashBeat Supabase access (for export)
- `SUPABASE_URL` and `SUPABASE_ANON_KEY` in `smashbeat/tools/.env`
- Never commit `.env` or raw ECG to git
- Supabase Free tier pauses after inactivity; no automatic backups

### HAWLEY ML needs (local only — no cloud secrets)
- ECGFounder base weights in `checkpoint/1_lead_ECGFounder.pth` (from Hugging Face)
- Hugging Face token only if re-downloading gated ECGFounder weights
- Supabase keys only if re-running a raw export (the current `exports/` archive is already local)

> **⚠️ Large assets are NOT in git — a fresh clone will not run.** Model weights and
> the ECGFounder base checkpoint are deliberately untracked (git host limit is 100 MB
> per file), so `checkpoint/` and most of `artifacts/` are empty after cloning.
> Before running anything, follow the bootstrap in
> [`docs/NOT_IN_GIT.md`](NOT_IN_GIT.md) — it lists every excluded path, its size,
> why it is excluded, and the exact command or download to restore it.
>
> Also note `exports/` (raw Sandbox ECG) is annotated elsewhere as gitignored but is
> **currently tracked** — see the discrepancy section in that document.

*(Former AWS credentials / SageMaker role / S3 bucket `hawley-ml-ecg` are defunct — account closed.)*

---

## 12. Regulatory & Product Framing

- **v1 positioning:** Personal cardiovascular wellness insights and trend tracking — **not a medical device**
- Sandbox tags are **not** clinical ground truth
- High-confidence pathology alerts require persistence + clean signal + human review pathway before any user-facing clinical language
- Explicit ML training consent needed before multi-user scale
- Consider IRB if research publication is a goal

---

## 13. Known Gaps & Open Questions

| Gap | Action for HAWLEY ML agent |
|-----|---------------------------|
| ~~Micah's exact `subject_id` in Supabase~~ | **Resolved:** `"Micah"` |
| `SBX-1783694914924` incomplete export (86 missing objects) | Re-run `export_sandbox.py --recording-id SBX-1783694914924` or exclude from Phase 0 |
| Duplicate export copies | HAWLEY ML canonical copy is `exports/`; SmashBeat may still have `smashbeat/exports/2026-07-16T162705Z/` |
| No link between Sandbox recordings and Live game sessions | Defer; optional schema extension in SmashBeat later |
| `tools/requirements.txt` lacks `zstandard` | Add to HAWLEY ML requirements |
| No ML code exists anywhere yet | Phase 0 is greenfield |
| ECGFounder size for edge deployment | Assume cloud-only until distillation Phase 5 |
| False-alert rate threshold for go/no-go | User to define; suggest starting with <5% on manually reviewed clean rest windows |

---

## 14. What the Agent Should NOT Do

1. **Do not modify SmashBeat Flutter app** unless explicitly asked — capture works
2. **Do not treat Sandbox tags as clinical labels** without validation subset
3. **Do not skip ECGFounder preprocessing** — results will not reproduce paper metrics
4. **Do not build on-device inference before Phase 0 passes**
5. **Do not use Supabase as long-term training storage** — the local `exports/` archive is the source of truth (AWS/S3 is gone)
6. **Do not implement full robotics RL** — use threshold adaptation + batch retraining
7. **Do not feed raw ECG waveforms to an LLM**
8. **Do not commit raw ECG, `.env`, or Supabase keys to git**
9. **Do not commit model weights or any single file over ~100 MB** — the push will be rejected. The excluded set (base checkpoint, `artifacts/**/*.pt`, bundle diagnosis head, `processed/**/*.npz`) is inventoried in [`docs/NOT_IN_GIT.md`](NOT_IN_GIT.md); add a row there and a `.gitignore` rule if you introduce a new large asset.

---

## 15. First Prompt for HAWLEY ML Agent

Suggested starting instruction:

> Read `docs/AGENT_CONTEXT.md` (this document). Phase 0 only: export Micah's Sandbox recordings from Supabase using the export workflow described in Section 4, implement an SBEC parser and zstd decompressor, build the 130→500 Hz preprocessing bridge matching ECGFounder's `dataset.py`, and run a feasibility report on clean rest windows: personal reconstruction error baseline and ECGFounder false non-normal alert rate. Do not modify SmashBeat. Do not build on-device inference. Output a go/no-go recommendation with numbers.

---

## 16. Conversation History Summary

Prior brainstorming (SmashBeat repo, 2026-07-15/16) established:
- ECGFounder + SageMaker + Supabase Sandbox data + optional LLM narrator
- Robotics feedback loop mapped to predictive coding (not action RL)
- Dual-goal: personal baseline primary, diagnosis-as-nominal-state secondary
- Separate repo for ML (HAWLEY ML) with S3 as training source of truth
- SmashBeat remains capture-only for foreseeable future

---

*Keep this file updated as phases complete.*
