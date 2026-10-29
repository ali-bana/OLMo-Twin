#!/usr/bin/env python3
"""Stage and verify the exact OLMo 2 1B Stage-2 training assets.

The published YAML remains authoritative. This utility maps its ordered HTTP
``data.paths`` list to deterministic files in a local directory,
downloads those raw uint32 token memmaps restartably, and writes a localized
YAML with only the paths changed. The files use a ``.npy`` suffix but do not
contain NumPy headers; this matches OLMo's ``MemMapDataset`` loader. The tool
deliberately does not accept the human-readable ``dolmino-mix-1124`` mirror as
trainer input.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

import numpy as np
import yaml


OLMO_COMMIT = "090253dac6688f2532509daa7aa2eb5fae50e956"
OFFICIAL_CONFIG_SHA256 = (
    "095dabb7b20c9b6710abad3717a789c9d3109e070069b8643805bb407d2a2fd1"
)
OFFICIAL_CONFIG_RELATIVE = Path(
    "configs/official-0425/OLMo2-1B-stage2-seed42.yaml"
)
CHECKPOINT_FILES = {
    "config.yaml": {
        "bytes": 163_724,
        "sha256": "ec6f24df31443597a5a281b09b03d46aba7f21f942d572bdff8d64fff18d21a3",
    },
    "train.pt": {
        "bytes": 15_500,
        "sha256": "7f5b2e840fa256f589dab6ae206e667e1ef4891d9c26ba1d4db364d6bdf45939",
    },
    "model.pt": {
        "bytes": 5_939_703_038,
        "sha256": "7b5a229bde2dc24a48fc269411e27c2bf3020446483a1e99a611d6985633b80b",
    },
    "optim.pt": {
        "bytes": 11_879_442_406,
        "sha256": "c5baee99495d0af7596cb58b69fd2ef108282bcf6348f02b8958ef9c6e41772a",
    },
}
MANIFEST_SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def load_official_config(
    path: Path, expected_sha256: str = OFFICIAL_CONFIG_SHA256
) -> dict[str, Any]:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"official config SHA-256 is {actual_sha256}; expected {expected_sha256}"
        )
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("official config is not a YAML mapping")
    return value


def authoritative_urls(config: dict[str, Any]) -> list[str]:
    try:
        paths = config["data"]["paths"]
    except (KeyError, TypeError) as exc:
        raise ValueError("official config has no data.paths list") from exc
    if not isinstance(paths, list) or not paths:
        raise ValueError("official config data.paths must be a non-empty list")

    urls: list[str] = []
    for index, value in enumerate(paths):
        if not isinstance(value, str):
            raise TypeError(f"data.paths[{index}] is not a string")
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"data.paths[{index}] is not HTTP(S): {value}")
        if parsed.hostname != "olmo-data.org":
            raise ValueError(f"unexpected training-data host in data.paths[{index}]")
        if not parsed.path.endswith(".npy"):
            raise ValueError(f"data.paths[{index}] is not an NPY shard: {value}")
        if parsed.query or parsed.fragment:
            raise ValueError(f"data.paths[{index}] contains query/fragment data")
        urls.append(value)
    if len(urls) != len(set(urls)):
        raise ValueError("official data.paths contains duplicate URLs")
    return urls


def url_relative_path(url: str) -> Path:
    relative = Path(urlsplit(url).path.lstrip("/"))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe URL path: {url}")
    return relative


def build_entries(urls: Iterable[str], data_root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    local_paths: set[str] = set()
    for index, url in enumerate(urls):
        relative = url_relative_path(url)
        local_path = (data_root / relative).resolve()
        if not local_path.is_relative_to(data_root.resolve()):
            raise ValueError(f"localized path escaped data root: {local_path}")
        local_text = str(local_path)
        if local_text in local_paths:
            raise ValueError(f"two URLs map to the same local path: {local_path}")
        local_paths.add(local_text)
        entries.append(
            {
                "index": index,
                "url": url,
                "relative_path": relative.as_posix(),
                "local_path": local_text,
                "status": "planned",
                "bytes": None,
                "tokens": None,
                "dtype": None,
                "sha256": None,
                "etag": None,
                "last_modified": None,
                "error": None,
            }
        )
    return entries


def merge_previous_entries(
    entries: list[dict[str, Any]], previous: dict[str, Any] | None
) -> list[dict[str, Any]]:
    if previous is None:
        return entries
    if previous.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("existing manifest has an unsupported schema version")
    if previous.get("olmo_commit") != OLMO_COMMIT:
        raise ValueError("existing manifest belongs to a different OLMo commit")
    if previous.get("official_config_sha256") != OFFICIAL_CONFIG_SHA256:
        raise ValueError("existing manifest belongs to a different official config")

    old_by_url = {entry["url"]: entry for entry in previous.get("entries", [])}
    if set(old_by_url) != {entry["url"] for entry in entries}:
        raise ValueError("existing manifest URL set differs from the official config")
    for entry in entries:
        old = old_by_url[entry["url"]]
        if old.get("index") != entry["index"]:
            raise ValueError("existing manifest changed the authoritative path order")
        if old.get("local_path") != entry["local_path"]:
            raise ValueError("existing manifest maps data into a different data root")
        for field in (
            "status",
            "bytes",
            "tokens",
            "dtype",
            "sha256",
            "etag",
            "last_modified",
            "error",
        ):
            entry[field] = old.get(field)
    return entries


def manifest_document(entries: list[dict[str, Any]], config_path: Path) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "updated_utc": utc_now(),
        "olmo_commit": OLMO_COMMIT,
        "official_config": str(config_path.resolve()),
        "official_config_sha256": OFFICIAL_CONFIG_SHA256,
        "path_count": len(entries),
        "complete_count": sum(entry["status"] == "complete" for entry in entries),
        "total_bytes": sum(int(entry.get("bytes") or 0) for entry in entries),
        "total_tokens": sum(int(entry.get("tokens") or 0) for entry in entries),
        "entries": entries,
    }


def write_localized_config(
    official: dict[str, Any], entries: list[dict[str, Any]], output_path: Path
) -> str:
    localized = copy.deepcopy(official)
    localized["data"]["paths"] = [entry["local_path"] for entry in entries]
    rendered = yaml.safe_dump(localized, sort_keys=False, width=1_000_000)
    atomic_text(output_path, rendered)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def plan_assets(args: argparse.Namespace) -> int:
    official = load_official_config(args.official_config)
    entries = build_entries(authoritative_urls(official), args.data_root)
    previous = None
    if args.manifest.is_file():
        previous = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = merge_previous_entries(entries, previous)
    manifest = manifest_document(entries, args.official_config)
    atomic_json(args.manifest, manifest)
    localized_sha = write_localized_config(official, entries, args.localized_config)
    provenance = {
        "generated_utc": utc_now(),
        "olmo_commit": OLMO_COMMIT,
        "official_config": str(args.official_config.resolve()),
        "official_config_sha256": OFFICIAL_CONFIG_SHA256,
        "localized_config": str(args.localized_config.resolve()),
        "localized_config_sha256": localized_sha,
        "only_semantic_change": "data.paths URLs mapped one-for-one to local files",
        "path_count": len(entries),
    }
    atomic_json(args.localized_config.with_suffix(".provenance.json"), provenance)
    print(json.dumps({**provenance, "manifest": str(args.manifest)}, indent=2))
    return 0


def head_one(entry: dict[str, Any], timeout: float, attempts: int) -> dict[str, Any]:
    error: str | None = None
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(
            entry["url"],
            method="HEAD",
            headers={"User-Agent": "olmo-twin-stager/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = copy.deepcopy(entry)
                content_length = response.headers.get("Content-Length")
                result["bytes"] = int(content_length) if content_length else None
                result["etag"] = response.headers.get("ETag")
                result["last_modified"] = response.headers.get("Last-Modified")
                result["error"] = None
                if result["status"] != "complete":
                    result["status"] = "available"
                return result
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        if attempt < attempts:
            time.sleep(min(2**attempt, 15))
    result = copy.deepcopy(entry)
    result["status"] = "probe_failed"
    result["error"] = error
    return result


def probe_assets(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = manifest["entries"]
    results: list[dict[str, Any] | None] = [None] * len(entries)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(head_one, entry, args.timeout, args.attempts): entry["index"]
            for entry in entries
        }
        for count, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            results[index] = future.result()
            if count % 25 == 0 or count == len(entries):
                current = [result for result in results if result is not None]
                print(f"probed {len(current)}/{len(entries)}", flush=True)
    complete = [result for result in results if result is not None]
    if len(complete) != len(entries):
        raise RuntimeError("probe did not return one result per entry")
    updated = manifest_document(complete, Path(manifest["official_config"]))
    atomic_json(args.manifest, updated)
    failures = [entry for entry in complete if entry["status"] == "probe_failed"]
    summary = {
        "paths": len(complete),
        "available": len(complete) - len(failures),
        "failures": len(failures),
        "known_bytes": sum(int(entry.get("bytes") or 0) for entry in complete),
    }
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


def validate_token_memmap(path: Path) -> tuple[int, str]:
    """Validate the headerless uint32 stream consumed by OLMo MemMapDataset."""
    byte_count = path.stat().st_size
    item_size = np.dtype(np.uint32).itemsize
    if byte_count <= 0:
        raise ValueError(f"{path} contains no tokens")
    if byte_count % item_size:
        raise ValueError(
            f"{path} has {byte_count} bytes; raw uint32 data must be divisible by "
            f"{item_size}"
        )
    array = np.memmap(path, mode="r", dtype=np.uint32)
    # Touch both ends so an inaccessible or truncated mapping fails acceptance.
    _ = int(array[0])
    _ = int(array[-1])
    return int(array.size), str(array.dtype)


def download_one(
    entry: dict[str, Any], attempts: int, timeout: float, rehash: bool
) -> dict[str, Any]:
    result = copy.deepcopy(entry)
    target = Path(result["local_path"])
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.is_file():
        expected_bytes = result.get("bytes")
        if expected_bytes is not None and target.stat().st_size != expected_bytes:
            target.unlink()
        else:
            tokens, dtype = validate_token_memmap(target)
            result.update(
                {
                    "status": "complete",
                    "bytes": target.stat().st_size,
                    "tokens": tokens,
                    "dtype": dtype,
                    "sha256": sha256_file(target)
                    if rehash or not result.get("sha256")
                    else result["sha256"],
                    "error": None,
                }
            )
            return result

    partial = target.with_suffix(target.suffix + ".part")
    command = [
        "curl",
        "--fail",
        "--location",
        "--retry",
        str(attempts),
        "--retry-all-errors",
        "--connect-timeout",
        str(int(timeout)),
        "--silent",
        "--show-error",
        "--continue-at",
        "-",
        "--output",
        str(partial),
        result["url"],
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"curl exited {completed.returncode} for {result['url']}")
    expected_bytes = result.get("bytes")
    if expected_bytes is not None and partial.stat().st_size != expected_bytes:
        raise ValueError(
            f"downloaded {partial.stat().st_size} bytes for {result['url']}; "
            f"HEAD reported {expected_bytes}"
        )
    tokens, dtype = validate_token_memmap(partial)
    digest = sha256_file(partial)
    os.replace(partial, target)
    result.update(
        {
            "status": "complete",
            "bytes": target.stat().st_size,
            "tokens": tokens,
            "dtype": dtype,
            "sha256": digest,
            "error": None,
        }
    )
    return result


def download_assets(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = manifest["entries"]
    by_index = {entry["index"]: entry for entry in entries}
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_one, entry, args.attempts, args.timeout, args.rehash
            ): entry
            for entry in entries
        }
        for count, future in enumerate(as_completed(futures), start=1):
            original = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # Preserve every failure in the restart manifest.
                result = copy.deepcopy(original)
                result["status"] = "download_failed"
                result["error"] = f"{type(exc).__name__}: {exc}"
                failures.append(result)
            by_index[result["index"]] = result
            ordered = [by_index[index] for index in range(len(entries))]
            atomic_json(
                args.manifest,
                manifest_document(ordered, Path(manifest["official_config"])),
            )
            print(
                f"finished {count}/{len(entries)}: "
                f"{result['status']} {result['relative_path']}",
                flush=True,
            )
    summary = {
        "paths": len(entries),
        "complete": sum(entry["status"] == "complete" for entry in by_index.values()),
        "failures": len(failures),
        "manifest": str(args.manifest),
    }
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


def git_head(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def verify_assets(args: argparse.Namespace) -> int:
    errors: list[str] = []
    try:
        official = load_official_config(args.official_config)
    except Exception as exc:
        official = None
        errors.append(f"official config: {exc}")

    try:
        commit = git_head(args.olmo_repo)
        if commit != OLMO_COMMIT:
            errors.append(f"OLMo checkout is {commit}, expected {OLMO_COMMIT}")
    except Exception as exc:
        errors.append(f"OLMo checkout: {exc}")

    if args.image is not None and (
        not args.image.is_file() or args.image.stat().st_size == 0
    ):
        errors.append(f"missing container image: {args.image}")

    for name, expected in CHECKPOINT_FILES.items():
        path = args.checkpoint_root / name
        if not path.is_file():
            errors.append(f"missing checkpoint file: {path}")
        elif path.stat().st_size != expected["bytes"]:
            errors.append(
                f"checkpoint {name} has {path.stat().st_size} bytes, "
                f"expected {expected['bytes']}"
            )
        elif sha256_file(path) != expected["sha256"]:
            errors.append(f"checkpoint {name} SHA-256 differs from the pinned file")
    if not args.checkpoint_alias.is_symlink():
        errors.append(f"checkpoint load alias is not a symlink: {args.checkpoint_alias}")
    else:
        if args.checkpoint_alias.resolve() != args.checkpoint_root.resolve():
            errors.append("checkpoint load alias points to the wrong directory")

    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        entries = manifest["entries"]
        if manifest.get("olmo_commit") != OLMO_COMMIT:
            errors.append("manifest records a different OLMo commit")
        if manifest.get("official_config_sha256") != OFFICIAL_CONFIG_SHA256:
            errors.append("manifest records a different official config")
    except Exception as exc:
        manifest = {}
        entries = []
        errors.append(f"manifest: {exc}")

    local_paths: list[str] = []
    total_tokens = 0
    total_bytes = 0
    for entry in entries:
        path = Path(entry["local_path"])
        local_paths.append(str(path))
        if entry.get("status") != "complete":
            errors.append(f"incomplete data shard {entry.get('index')}: {entry.get('status')}")
            continue
        if not path.is_file():
            errors.append(f"missing data shard: {path}")
            continue
        actual_bytes = path.stat().st_size
        if actual_bytes != entry.get("bytes"):
            errors.append(f"data shard size changed: {path}")
            continue
        try:
            actual_tokens, actual_dtype = validate_token_memmap(path)
        except Exception as exc:
            errors.append(f"invalid data shard {path}: {exc}")
            continue
        if actual_tokens != entry.get("tokens") or actual_dtype != entry.get("dtype"):
            errors.append(f"data shard metadata changed: {path}")
            continue
        if not entry.get("sha256"):
            errors.append(f"data shard has no recorded SHA-256: {path}")
            continue
        if args.hash_data and sha256_file(path) != entry.get("sha256"):
            errors.append(f"data shard hash changed: {path}")
            continue
        total_bytes += actual_bytes
        total_tokens += int(entry.get("tokens") or 0)

    if official is not None:
        expected_urls = authoritative_urls(official)
        expected_count = len(expected_urls)
        if len(entries) != expected_count:
            errors.append(
                f"manifest has {len(entries)} paths; official config has {expected_count}"
            )
        if [entry.get("index") for entry in entries] != list(range(len(entries))):
            errors.append("manifest indices are not contiguous and authoritative")
        if [entry.get("url") for entry in entries] != expected_urls:
            errors.append("manifest URL order differs from official data.paths")
    try:
        localized = yaml.safe_load(args.localized_config.read_text(encoding="utf-8"))
        if localized["data"]["paths"] != local_paths:
            errors.append("localized config changed data path order or membership")
        if official is not None:
            expected_localized = copy.deepcopy(official)
            expected_localized["data"]["paths"] = local_paths
            if localized != expected_localized:
                errors.append(
                    "localized config changed settings other than data.paths"
                )
    except Exception as exc:
        errors.append(f"localized config: {exc}")

    if total_tokens < 50_000_000_000:
        errors.append(
            f"localized shards expose only {total_tokens:,} uint32 tokens; "
            "the published run requires 50B"
        )

    report = {
        "schema_version": 1,
        "verified_utc": utc_now(),
        "status": "ok" if not errors else "error",
        "errors": errors,
        "olmo_commit": OLMO_COMMIT,
        "official_config_sha256": OFFICIAL_CONFIG_SHA256,
        "path_count": len(entries),
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
        "hash_data": args.hash_data,
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "checkpoint_alias": str(args.checkpoint_alias),
        "image": None
        if args.image is None
        else (str(args.image.resolve()) if args.image.exists() else str(args.image)),
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="map official URLs to local paths")
    plan.add_argument("--official-config", type=Path, required=True)
    plan.add_argument("--data-root", type=Path, required=True)
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--localized-config", type=Path, required=True)
    plan.set_defaults(function=plan_assets)

    probe = subparsers.add_parser("probe", help="HEAD every URL to estimate storage")
    probe.add_argument("--manifest", type=Path, required=True)
    probe.add_argument("--workers", type=int, default=16)
    probe.add_argument("--timeout", type=float, default=30)
    probe.add_argument("--attempts", type=int, default=5)
    probe.set_defaults(function=probe_assets)

    download = subparsers.add_parser(
        "download", help="download every raw token-memmap shard restartably"
    )
    download.add_argument("--manifest", type=Path, required=True)
    download.add_argument("--workers", type=int, default=8)
    download.add_argument("--timeout", type=float, default=30)
    download.add_argument("--attempts", type=int, default=12)
    download.add_argument("--rehash", action="store_true")
    download.set_defaults(function=download_assets)

    verify = subparsers.add_parser("verify", help="fail closed unless all assets match")
    verify.add_argument("--olmo-repo", type=Path, required=True)
    verify.add_argument("--official-config", type=Path, required=True)
    verify.add_argument("--localized-config", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--checkpoint-root", type=Path, required=True)
    verify.add_argument("--checkpoint-alias", type=Path, required=True)
    verify.add_argument(
        "--image",
        type=Path,
        default=None,
        help="optional container image file; checked only when given",
    )
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--hash-data", action="store_true")
    verify.set_defaults(function=verify_assets)
    return root


def main() -> int:
    args = parser().parse_args()
    if hasattr(args, "workers") and args.workers <= 0:
        raise SystemExit("--workers must be positive")
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
