"""download_data.verify must check the PROVENANCE manifests where snapshot_download puts them.

Manifest paths are relative to the Hub repo root, which is mirrored into ``--dest`` (``data/``),
so the check has to read both manifests from ``dest`` and run ``shasum`` with ``cwd=dest`` —
previously it looked at the repo root and silently skipped.
"""
import hashlib
import importlib.util
import shutil
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "download_data.py"
_spec = importlib.util.spec_from_file_location("download_data", _SCRIPT)
download_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(download_data)

pytestmark = pytest.mark.skipif(shutil.which("shasum") is None, reason="needs shasum")


def _write_manifest(manifest: Path, root: Path, rel_paths: list[str]) -> None:
    lines = ["# header line is not a shasum entry"]
    for rel in rel_paths:
        digest = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        lines.append(f"{digest}  {rel}")
    manifest.write_text("\n".join(lines) + "\n")


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    """A fake Hub snapshot: main manifest at the root, Mode B manifest under data/mode_b/."""
    dest = tmp_path / "data"
    (dest / "shared_corpus" / "crm").mkdir(parents=True)
    (dest / "shared_corpus" / "crm" / "doc.pdf").write_bytes(b"%PDF-fake")
    (dest / "eval.json").write_text("[]")
    _write_manifest(dest / "PROVENANCE_hashes.txt", dest,
                    ["eval.json", "shared_corpus/crm/doc.pdf"])
    (dest / "data" / "mode_b" / "malicious").mkdir(parents=True)
    (dest / "data" / "mode_b" / "malicious" / "x.png").write_bytes(b"png")
    _write_manifest(dest / "data" / "mode_b" / "MODE_B_PROVENANCE_hashes.txt", dest,
                    ["data/mode_b/malicious/x.png"])
    return dest


def test_verify_passes_on_intact_snapshot(snapshot: Path):
    assert download_data.verify(snapshot) == 0


def test_verify_fails_on_tampered_corpus_file(snapshot: Path):
    (snapshot / "shared_corpus" / "crm" / "doc.pdf").write_bytes(b"tampered")
    assert download_data.verify(snapshot) == 1


def test_verify_checks_mode_b_manifest(snapshot: Path):
    (snapshot / "data" / "mode_b" / "malicious" / "x.png").write_bytes(b"tampered")
    assert download_data.verify(snapshot) == 1


def test_verify_reports_missing_manifest(tmp_path: Path, capsys):
    assert download_data.verify(tmp_path) == 0
    assert "No PROVENANCE manifest found" in capsys.readouterr().out
