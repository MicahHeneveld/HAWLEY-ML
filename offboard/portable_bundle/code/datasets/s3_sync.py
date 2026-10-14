"""Upload local Sandbox exports to S3 (Phase 1 data platform).

Credentials are read from the environment only — never from repo files.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets.manifest import write_manifest
from datasets.s3_io import ensure_bucket, s3_client, upload_tree


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync Sandbox export to S3")
    parser.add_argument(
        "--export-dir",
        default="exports/2026-07-16T162705Z",
        help="Local export directory",
    )
    parser.add_argument("--subject-id", default="Micah")
    parser.add_argument(
        "--bucket",
        default=os.environ.get("HAWLEY_ML_S3_BUCKET", "hawley-ml-ecg"),
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="S3 prefix (default: raw/{subject_id})",
    )
    parser.add_argument(
        "--create-bucket",
        action="store_true",
        help="Create bucket if it does not exist",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Skip files already on S3 with matching size",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build manifest only, do not upload",
    )
    args = parser.parse_args()

    export_dir = Path(args.export_dir).resolve()
    if not export_dir.is_dir():
        raise SystemExit(f"Export dir not found: {export_dir}")

    manifest_path = export_dir / "manifest_v1.json"
    print("Building manifest...", flush=True)
    write_manifest(export_dir, manifest_path, subject_id=args.subject_id)
    print(f"Wrote manifest: {manifest_path}")

    if args.dry_run:
        print("Dry run — skipping S3 upload.")
        return

    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        raise SystemExit(
            "AWS credentials not set. Export AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY "
            "in your shell or a local .env file (never commit .env)."
        )

    client = s3_client()
    if args.create_bucket:
        ensure_bucket(client, args.bucket)

    prefix = args.prefix or f"raw/{args.subject_id.lower()}"
    uploaded, skipped = upload_tree(
        client,
        args.bucket,
        export_dir,
        prefix,
        incremental=args.incremental,
    )
    manifest_key = f"{prefix}/manifest_v1.json"
    client.upload_file(str(manifest_path), args.bucket, manifest_key)
    print(
        f"Uploaded {uploaded} files ({skipped} skipped) to s3://{args.bucket}/{prefix}/"
    )


if __name__ == "__main__":
    main()
