import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import yaml


MIDTRAIN = Path(__file__).resolve().parents[1]  # training/midtrain
SCRIPT = MIDTRAIN / "prepare_stage2_assets.py"
SPEC = importlib.util.spec_from_file_location("prepare_stage2_assets", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def sample_config():
    return {
        "run_name": "test",
        "model": {"max_sequence_length": 4096},
        "data": {
            "memmap_dtype": "uint32",
            "paths": [
                "http://olmo-data.org/preprocessed/a/dolma2-tokenizer/part-1.npy",
                "https://olmo-data.org/preprocessed/b/dolma2-tokenizer/part-2.npy",
            ],
        },
    }


def test_official_config_hash_is_enforced(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(sample_config(), sort_keys=False), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert MODULE.load_official_config(path, digest) == sample_config()

    try:
        MODULE.load_official_config(path, "0" * 64)
    except ValueError as exc:
        assert "SHA-256" in str(exc)
    else:
        raise AssertionError("a changed official config was accepted")


def test_paths_are_localized_one_for_one_in_authoritative_order(tmp_path):
    config = sample_config()
    urls = MODULE.authoritative_urls(config)
    entries = MODULE.build_entries(urls, tmp_path / "data")

    assert [entry["index"] for entry in entries] == [0, 1]
    assert [entry["url"] for entry in entries] == urls
    assert entries[0]["relative_path"] == (
        "preprocessed/a/dolma2-tokenizer/part-1.npy"
    )

    localized_path = tmp_path / "localized.yaml"
    MODULE.write_localized_config(config, entries, localized_path)
    localized = yaml.safe_load(localized_path.read_text(encoding="utf-8"))
    assert localized["data"]["paths"] == [entry["local_path"] for entry in entries]
    assert localized["model"] == config["model"]
    assert localized["data"]["memmap_dtype"] == "uint32"


def test_previous_manifest_cannot_change_order_or_data_root(tmp_path):
    entries = MODULE.build_entries(
        MODULE.authoritative_urls(sample_config()), tmp_path / "data"
    )
    previous = MODULE.manifest_document(entries, tmp_path / "official.yaml")
    previous["entries"][0]["status"] = "complete"
    previous["entries"][0]["bytes"] = 123
    merged = MODULE.merge_previous_entries(
        MODULE.build_entries(
            MODULE.authoritative_urls(sample_config()), tmp_path / "data"
        ),
        previous,
    )
    assert merged[0]["status"] == "complete"
    assert merged[0]["bytes"] == 123

    changed = json.loads(json.dumps(previous))
    changed["entries"][0]["index"] = 1
    try:
        MODULE.merge_previous_entries(entries, changed)
    except ValueError as exc:
        assert "order" in str(exc)
    else:
        raise AssertionError("a reordered manifest was accepted")


def test_token_memmap_validation_requires_nonempty_uint32_aligned_data(tmp_path):
    valid = tmp_path / "valid.npy"
    np.asarray([1, 2, 3], dtype=np.uint32).tofile(valid)
    assert MODULE.validate_token_memmap(valid) == (3, "uint32")

    invalid = tmp_path / "invalid.npy"
    invalid.write_bytes(b"not-aligned")
    try:
        MODULE.validate_token_memmap(invalid)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("a byte-misaligned training shard was accepted")

    empty = tmp_path / "empty.npy"
    empty.touch()
    try:
        MODULE.validate_token_memmap(empty)
    except ValueError as exc:
        assert "no tokens" in str(exc)
    else:
        raise AssertionError("an empty training shard was accepted")


def test_checkpoint_identity_includes_pinned_hashes():
    assert set(MODULE.CHECKPOINT_FILES) == {
        "config.yaml",
        "train.pt",
        "model.pt",
        "optim.pt",
    }
    for identity in MODULE.CHECKPOINT_FILES.values():
        assert identity["bytes"] > 0
        assert len(identity["sha256"]) == 64
        int(identity["sha256"], 16)


def test_download_accepts_existing_headerless_token_memmap(tmp_path):
    path = tmp_path / "tokens.npy"
    values = np.asarray([7, 11, 13, 17], dtype=np.uint32)
    values.tofile(path)
    entry = {
        "index": 0,
        "url": "https://olmo-data.org/preprocessed/test/tokens.npy",
        "relative_path": "preprocessed/test/tokens.npy",
        "local_path": str(path),
        "status": "available",
        "bytes": path.stat().st_size,
        "tokens": None,
        "dtype": None,
        "sha256": None,
        "etag": None,
        "last_modified": None,
        "error": None,
    }

    result = MODULE.download_one(entry, attempts=1, timeout=1, rehash=False)

    assert result["status"] == "complete"
    assert result["tokens"] == len(values)
    assert result["dtype"] == "uint32"
    assert result["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_checkpoint_downloader_uses_the_same_pinned_identity():
    downloader = (MIDTRAIN / "download_stage1_checkpoint.sh").read_text(
        encoding="utf-8"
    )
    for name, identity in MODULE.CHECKPOINT_FILES.items():
        assert name in downloader
        assert str(identity["bytes"]) in downloader
        assert identity["sha256"] in downloader
