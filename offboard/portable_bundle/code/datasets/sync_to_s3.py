"""Push HAWLEY ML trees to S3 (raw export, artifacts, reports, processed)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets.s3_io import ensure_bucket, s3_client, upload_tree


def _require_creds() -> None:
    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        raise SystemExit(
            "AWS credentials not set. Use SSO/env vars locally (never commit .env)."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload HAWLEY ML paths to S3")
    parser.add_argument(
        "--bucket",
        default=os.environ.get("HAWLEY_ML_S3_BUCKET", "hawley-ml-ecg"),
    )
    parser.add_argument("--create-bucket", action="store_true")
    parser.add_argument(
        "--target",
        choices=("artifacts", "reports", "processed", "all"),
        default="artifacts",
    )
    parser.add_argument(
        "--local-root",
        default=None,
        help="Override local directory (default depends on --target)",
    )
    parser.add_argument(
        "--s3-prefix",
        default=None,
        help="Override S3 prefix (default depends on --target)",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Skip files that already exist with matching size",
    )
    args = parser.parse_args()

    defaults = {
        "artifacts": ("artifacts", "artifacts"),
        "reports": ("reports", "reports"),
        "processed": ("processed", "processed"),
    }
    if args.target == "all":
        targets = ("artifacts", "reports", "processed")
    else:
        targets = (args.target,)

    _require_creds()
    client = s3_client()
    if args.create_bucket:
        ensure_bucket(client, args.bucket)

    for target in targets:
        local_default, prefix_default = defaults[target]
        local_root = Path(args.local_root or local_default).resolve()
        prefix = args.s3_prefix or prefix_default
        if not local_root.is_dir():
            print(f"Skip {target}: missing {local_root}")
            continue
        print(f"Uploading {local_root} -> s3://{args.bucket}/{prefix}/", flush=True)
        up, skip = upload_tree(
            client,
            args.bucket,
            local_root,
            prefix,
            incremental=args.incremental,
        )
        print(f"Done {target}: {up} uploaded, {skip} skipped.")


if __name__ == "__main__":
    main()
