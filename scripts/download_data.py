#!/usr/bin/env python3
# Copyright (c) 2026, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fetch the ErosionBench dataset + companion corpus from Hugging Face into ``data/``.

The dataset JSONs and the PDF corpus are NOT committed to git (they are large, frozen
binary artifacts); they live in a Hugging Face dataset repo and are pulled on demand.

Usage
-----
    python scripts/download_data.py                     # default repo, into ./data
    python scripts/download_data.py --repo-id ORG/NAME  # override the HF repo
    python scripts/download_data.py --revision v1.1     # pin a tag/commit
    python scripts/download_data.py --verify            # re-hash against PROVENANCE_hashes.txt

The repo id can also be set with the ``EROSIONBENCH_HF_REPO`` environment variable.
After download, integrity can be checked against the pinned SHA-256 in ``DATASET_CARD.md``.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

# The published dataset repo. Override with --repo-id or EROSIONBENCH_HF_REPO until this
# is finalized on the Hub.
DEFAULT_REPO_ID = os.environ.get("EROSIONBENCH_HF_REPO", "Salesforce/ErosionBench")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

# Pinned integrity anchor for the full pool (see DATASET_CARD.md changelog, v1.1).
PINNED_DATASET_SHA256 = "faf765c73a071d70305b11e49caaca9317fc88398e89a7891039e9218bc0e3bf"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(repo_id: str, revision: str | None, dest: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(
            "huggingface_hub is required. Install it with:\n"
            "    pip install huggingface_hub\n"
            "(it is also pulled in by `pip install -e .`)."
        )

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {repo_id!r}"
          + (f" @ {revision}" if revision else "")
          + f" -> {dest} ...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(dest),
        # The Hub README is the dataset card (tracked here as DATASET_CARD.md); skipping it keeps
        # the tracked data/README.md intact. It is not in either PROVENANCE manifest. parquet/ is
        # the derived load_dataset view of the split JSONs; the harness reads the JSON, so skip it.
        ignore_patterns=["README.md", ".gitattributes", "parquet/*"],
    )
    print("Download complete.")


def verify(dest: Path) -> int:
    """Check the full-pool hash and, if present, run the PROVENANCE manifest check."""
    ok = True
    dataset_json = dest / "dataset.json"
    if dataset_json.exists():
        got = _sha256(dataset_json)
        match = got == PINNED_DATASET_SHA256
        print(f"dataset.json SHA-256: {got} "
              f"{'OK (matches pinned v1.1)' if match else 'MISMATCH — expected ' + PINNED_DATASET_SHA256}")
        ok = ok and match
    else:
        print(f"dataset.json not found in {dest} — skipping hash check.")

    # Manifest paths are relative to the Hub repo root, which snapshot_download mirrors into
    # `dest` — so both manifests live under `dest` and must be checked from there.
    manifests = [dest / "PROVENANCE_hashes.txt", dest / "data" / "mode_b" / "MODE_B_PROVENANCE_hashes.txt"]
    found = [m for m in manifests if m.exists()]
    if not found:
        print(f"No PROVENANCE manifest found in {dest} — skipping manifest check.")
    for manifest in found:
        print(f"Running manifest check against {manifest.relative_to(dest)} ...")
        # Only the lines that begin with a hex hash are shasum entries.
        lines = [ln for ln in manifest.read_text().splitlines() if ln[:1].isalnum() and " " in ln]
        proc = subprocess.run(
            ["shasum", "-a", "256", "-c", "-"],
            input="\n".join(lines).encode(),
            cwd=str(dest),
        )
        ok = ok and proc.returncode == 0

    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                    help=f"Hugging Face dataset repo id (default: {DEFAULT_REPO_ID})")
    ap.add_argument("--revision", default=None, help="tag / branch / commit to pin (e.g. v1.1)")
    ap.add_argument("--dest", default=str(DATA_DIR), help="destination directory (default: ./data)")
    ap.add_argument("--verify", action="store_true", help="verify integrity after download")
    ap.add_argument("--verify-only", action="store_true", help="skip download, only verify existing data/")
    args = ap.parse_args()

    dest = Path(args.dest)
    if not args.verify_only:
        download(args.repo_id, args.revision, dest)

    if args.verify or args.verify_only:
        return verify(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
