# Data

The dataset JSONs and PDF corpus are **not committed to this repository** — they are hosted on
Hugging Face and fetched on demand. This keeps the git repo small and makes the frozen, versioned
artifact the single source of truth.

## Fetch

```bash
python scripts/download_data.py
```

This populates the following into `data/`:

| Path | What |
|---|---|
| `data/dataset.json` | Full 1,728-scenario pool (superset; **not** a clean test set) |
| `data/calib.json` | Calibration split — 1,124 scenarios, 281 families |
| `data/eval.json` | **Held-out** evaluation split — 604 scenarios, 151 families |
| `data/eval_clean_control.json` | **Held-out** clean-control arm (paper headline) — `eval.json`'s 453 malicious + 453 validity-matched benign controls, same 151 families |
| `data/shared_corpus/` | 32 base PDFs (8 docs × 4 domains) + per-domain `corpus_manifest.json` |
| `data/verification_manifest.json` | Recovered-payload capturability table |
| `data/PROVENANCE_hashes.txt` | SHA-256 of every file above (42 files) |
| `data/data/mode_b/` | Multimodal (Mode B) subset: 66 malicious + 66 provisional benign image scenarios, with `MODE_B_PROVENANCE_hashes.txt` |

The Hub repo root is mirrored into `data/`, so the Hub's own `data/mode_b/` folder lands at
`data/data/mode_b/`.

> **Note:** the frozen calibration artifacts are **not** part of the download — they are tracked in
> [`../artifacts/`](../artifacts/) (`artifact.json` = auditor `claude-opus-4-8`, τ = 0.595). Each is
> **auditor-specific**: to evaluate an auditor without a shipped artifact, fit one with
> `scripts/collect_calibration.py` on `calib.json` rather than reusing another auditor's artifact.

## Integrity

After fetching, verify the frozen bytes (manifest paths are relative to `data/`):

```bash
python scripts/download_data.py --verify-only        # checks both manifests
# or by hand, from the repo root:
(cd data && grep '^[0-9a-f]' PROVENANCE_hashes.txt | shasum -a 256 -c -)
(cd data && grep '^[0-9a-f]' data/mode_b/MODE_B_PROVENANCE_hashes.txt | shasum -a 256 -c -)
```

The PDF corpus is **LLM-generated and not byte-reproducible from the seed** — the PDFs are frozen,
required binary artifacts. Do not regenerate them expecting identical bytes.

## Version

Current: **v1.1** — see the [changelog](../DATASET_CARD.md#changelog).
Pinned `dataset.json` SHA-256: `faf765c73a071d70305b11e49caaca9317fc88398e89a7891039e9218bc0e3bf`.
