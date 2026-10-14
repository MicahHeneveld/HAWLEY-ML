# SmashBeat sandbox export

This folder is a local archive exported from the SmashBeat Supabase sandbox on
2026-07-16. It contains 32 recording directories and a batch-level summary.

## Folder structure

```text
2026-07-16T162705Z/
|-- README.md
|-- batch_summary.json
`-- SBX-<recording-id>/
    |-- recording.json
    |-- chunks.csv
    |-- export_summary.json
    `-- chunks/
        |-- chunk_00000.bin.zst
        |-- chunk_00000_beats.jsonl.zst
        `-- ...
```

- `batch_summary.json`: export status and statistics for every recording.
- `recording.json`: recording-level metadata such as subject, device, times,
  status, and schema version.
- `chunks.csv`: manifest mapping each database chunk row to its local files,
  checksums, compression, timestamps, and download status.
- `export_summary.json`: per-recording counts, byte totals, missing-object list,
  and completion status.
- `chunks/*.bin.zst`: Zstandard-compressed raw ECG binary chunks.
- `chunks/*_beats.jsonl.zst`: Zstandard-compressed beat/tag records in JSON
  Lines format.

Paths in the JSON and CSV metadata may still reference the original export
location under `C:\SmashBeat`; use the relative `bin_local` and `beats_local`
columns in `chunks.csv` to locate files within this copy.

## Archive status

This copy contains 20,967 compressed chunk files across 31 complete recordings.
Six recording records have zero chunk rows, so their `chunks` directories are empty.

The incomplete recording `SBX-1783694914924` was removed from this HAWLEY ML copy.
`batch_summary.json` reflects 31 complete recordings with `"all_ok": true`.

## Subject

All recordings in this export use `subject_id: "Micah"`.
