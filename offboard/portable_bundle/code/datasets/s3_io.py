"""Shared S3 upload/download helpers."""

from __future__ import annotations

import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


def s3_client():
    return boto3.client(
        "s3",
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"),
    )


def ensure_bucket(client, bucket: str) -> None:
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError:
        region = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
        if region == "us-east-1":
            client.create_bucket(Bucket=bucket)
        else:
            client.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": region},
            )


def _local_size(path: Path) -> int:
    return path.stat().st_size


def _should_skip_upload(client, bucket: str, key: str, local_path: Path) -> bool:
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except ClientError:
        return False
    remote_size = head.get("ContentLength")
    return remote_size == _local_size(local_path)


def upload_tree(
    client,
    bucket: str,
    local_root: Path,
    s3_prefix: str,
    *,
    incremental: bool = False,
) -> tuple[int, int]:
    """Returns (uploaded_count, skipped_count)."""
    uploaded = 0
    skipped = 0
    paths = [path for path in local_root.rglob("*") if path.is_file()]
    total = len(paths)
    for path in paths:
        key = f"{s3_prefix}/{path.relative_to(local_root).as_posix()}"
        if incremental and _should_skip_upload(client, bucket, key, path):
            skipped += 1
            done = uploaded + skipped
            if done % 500 == 0 or done == total:
                print(
                    f"S3 upload: {uploaded} uploaded, {skipped} skipped ({done}/{total})",
                    flush=True,
                )
            continue
        client.upload_file(str(path), bucket, key)
        uploaded += 1
        done = uploaded + skipped
        if done % 500 == 0 or done == total:
            print(
                f"S3 upload: {uploaded} uploaded, {skipped} skipped ({done}/{total})",
                flush=True,
            )
    return uploaded, skipped


def upload_file(client, bucket: str, local_path: Path, key: str) -> None:
    client.upload_file(str(local_path), bucket, key)


def download_prefix(
    client,
    bucket: str,
    s3_prefix: str,
    local_root: Path,
) -> int:
    """Download all keys under prefix. Returns file count."""
    local_root.mkdir(parents=True, exist_ok=True)
    prefix = s3_prefix.rstrip("/") + "/"
    paginator = client.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key[len(prefix) :]
            dest = local_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(dest))
            count += 1
            if count % 500 == 0:
                print(f"S3 download: {count} files...", flush=True)
    return count


def list_top_level_prefixes(client, bucket: str) -> list[str]:
    resp = client.list_objects_v2(Bucket=bucket, Delimiter="/")
    return [p["Prefix"].rstrip("/") for p in resp.get("CommonPrefixes") or []]
