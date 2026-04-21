"""Small MinIO/S3 helpers used by ClearML agent wrappers.

The source WSI files live in MinIO and are not registered as a ClearML Dataset.
Credentials are intentionally read from environment variables on the agent.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import boto3
from botocore.config import Config


SUPPORTED_WSI_EXTENSIONS = (".tif", ".tiff", ".svs", ".ndpi")


def bool_param(value: object, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def minio_config(params: dict) -> dict:
    """Read MinIO config from ClearML params plus agent environment variables."""
    return {
        "endpoint": str(
            params.get("minio_endpoint")
            or os.environ.get("MINIO_ENDPOINT")
            or "https://api.blackhole2.ai.innopolis.university:443"
        ),
        "bucket": str(params.get("minio_bucket") or os.environ.get("MINIO_BUCKET") or "pershin-medailab"),
        "prefix": str(
            params.get("minio_prefix")
            or os.environ.get("MINIO_PREFIX")
            or "Pathomorphology/CAMELYON/16/training"
        ).strip("/"),
        "access_key": (
            os.environ.get("MINIO_ACCESS_KEY")
            or os.environ.get("AWS_ACCESS_KEY_ID")
            or str(params.get("minio_access_key") or "")
        ),
        "secret_key": (
            os.environ.get("MINIO_SECRET_KEY")
            or os.environ.get("AWS_SECRET_ACCESS_KEY")
            or str(params.get("minio_secret_key") or "")
        ),
        "verify_ssl": bool_param(params.get("verify_ssl"), True),
    }


def get_minio_client(config: dict):
    if not config["access_key"] or not config["secret_key"]:
        raise RuntimeError(
            "MinIO credentials are missing. Set MINIO_ACCESS_KEY and MINIO_SECRET_KEY "
            "in the ClearML agent environment."
        )

    return boto3.client(
        "s3",
        endpoint_url=config["endpoint"],
        aws_access_key_id=config["access_key"],
        aws_secret_access_key=config["secret_key"],
        region_name="us-east-1",
        verify=config["verify_ssl"],
        config=Config(s3={"addressing_style": "path"}),
    )


def join_prefix(*parts: str) -> str:
    return "/".join(str(part).strip("/") for part in parts if str(part).strip("/"))


def list_slide_keys(
    client,
    bucket: str,
    prefix: str,
    extensions: Iterable[str] = SUPPORTED_WSI_EXTENSIONS,
) -> list[str]:
    prefix = prefix.strip("/") + "/"
    allowed = tuple(ext.lower() for ext in extensions)
    keys: list[str] = []

    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(allowed):
                keys.append(key)

    return sorted(keys)


def list_keys(
    client,
    bucket: str,
    prefix: str,
    suffix: str | tuple[str, ...] | None = None,
) -> list[str]:
    prefix = prefix.strip("/") + "/"
    keys: list[str] = []

    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if suffix is None or key.lower().endswith(suffix):
                keys.append(key)

    return sorted(keys)


def slide_key_by_stem(keys: Iterable[str]) -> dict[str, str]:
    return {Path(key).stem: key for key in keys}


def object_exists(client, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except client.exceptions.ClientError as exc:
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status == 404:
            return False
        raise


def download_slide(client, bucket: str, key: str, local_dir: Path) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / Path(key).name
    if local_path.exists() and local_path.stat().st_size > 0:
        print(f"[CACHE] {local_path}")
        return local_path

    print(f"[DOWNLOAD] s3://{bucket}/{key} -> {local_path}")
    client.download_file(bucket, key, str(local_path))
    return local_path


def download_key(client, bucket: str, key: str, local_path: Path) -> Path:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists() and local_path.stat().st_size > 0:
        print(f"[CACHE] {local_path}")
        return local_path

    print(f"[DOWNLOAD] s3://{bucket}/{key} -> {local_path}")
    client.download_file(bucket, key, str(local_path))
    return local_path


def upload_file(client, bucket: str, local_path: Path, key: str, overwrite: bool = False) -> None:
    if not overwrite and object_exists(client, bucket, key):
        print(f"[SKIP] Existing MinIO object: s3://{bucket}/{key}")
        return

    print(f"[UPLOAD] {local_path} -> s3://{bucket}/{key}")
    client.upload_file(str(local_path), bucket, key)
