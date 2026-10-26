# ErosionBench — Release Artifact

A benchmark for measuring **alignment decay**: how attack success against a multi-agent LLM
mesh escalates as coordination increases (single agent → no-defense mesh → perimeter/privilege
defenses → calibrated Meta-Auditor), across four enterprise domains and 16 attack classes.

This artifact is **self-contained and offline** for the environment: the sandbox needs no LLM
keys. Reproducing the *scores* (running a mesh + auditor over the scenarios) needs model access.

---

## What's in the release

The dataset, corpus, and provenance files are hosted on the Hugging Face Hub at
[`Salesforce/ErosionBench`](https://huggingface.co/datasets/Salesforce/ErosionBench) and fetched into
`data/` with `python3 scripts/download_data.py`; the sandbox ships in this repository. Paths below are
relative to the repository root after that download.

| Part | Path | What it is |
|---|---|---|
| **1. Dataset** | `data/{dataset,calib,eval}.json` | 1,728-scenario pool + family-disjoint calibration (1,124) / held-out eval (604) splits |
| **2. Corpus** | `data/shared_corpus/` | 32 clean base PDFs (8 docs × 4 domains) + per-domain `corpus_manifest.json`. The PDFs carry no payloads; each malicious scenario's needle is spliced into its target page by the sandbox at run time. **Required, non-reproducible binary artifacts.** |
| **3. Sandbox** | `erosionbench/sandbox/` | Offline FastAPI HTTP tool server the agents call (enterprise actions + document API + telemetry) |
| **4. Provenance** | `data/verification_manifest.json`, `data/PROVENANCE_hashes.txt` | Corpus content hashes + recovered-payload table (all 4 domains); SHA-256 of every shipped file |

### Splits (HuggingFace mapping)
- `calib.json` → `calibration` / `train` split (1,124 scenarios, 281 families)
- `eval.json` → `evaluation` / `test` split (604 scenarios, 151 families) — **the held-out paper number**
- **Family-disjoint**: no base/`_CONTROL`/`_T2`/`_T3` variant of a family straddles the split
  boundary (verified: overlap = 0). This preserves the split-conformal calibration guarantee.
- **3:1 malicious:benign**, with a benign `_CONTROL` twin per family for over-refusal (FPR) measurement.
- **Tiers** T1 (base) / T2 / T3 encode escalating adversarial conditioning of the *same* contract —
  the alignment-decay ramp (attack success rises T1 < T2 < T3).

### Multimodal (Mode B) — image evaluation subset (included)

The paper also reports a **multimodal (Mode B)** arm (attacks delivered as rendered document
images, testing channel-invariant detection). The Hugging Face release **includes** the Mode B image
subset under its `data/mode_b/` folder (fetched to `data/data/mode_b/` here): **66 malicious**
(`malicious/`) + **66 provisional benign-twin** controls (`benign_provisional/`) — 11 attack classes ×
2 families × 3 tiers — each a deterministic, refusal-immune freeze pairing an `image` carrier (payload
in a PNG, recoverable only from the image) with a `text` carrier. A zero-download 2 + 2 sample ships
in this repo under `data/sample/mode_b/`. Verify the full subset with (manifest paths are relative to
the Hub repo root, i.e. `data/`):
```bash
(cd data && grep '^[0-9a-f]' data/mode_b/MODE_B_PROVENANCE_hashes.txt | shasum -a 256 -c -)
```
⚠️ The Mode B **benign controls are provisional** — not yet validity-matched authorized completions,
so **do not compute a Mode B FPR** from them. The paper reports Mode B as channel-invariant detection
plus a valid-control frontier, and does **not** claim it reproduces the Mode A text FPR. Validity-matched
benign controls replace that folder in a future version; the malicious images stay byte-identical.
See `DATASET_CARD.md` → *Multimodal (Mode B)* and the paper's Mode B appendix.

---

## Quickstart — run the sandbox

### Option A: Docker (recommended)
```bash
python3 scripts/download_data.py      # fetches data/shared_corpus/, which the image bakes in
docker build -f Dockerfile.sandbox -t erosionbench-sandbox .
docker run --rm -p 8199:8199 erosionbench-sandbox
```
No download yet? Build against the bundled 12-doc sample corpus instead:
`docker build -f Dockerfile.sandbox --build-arg CORPUS_DIR=data/sample/shared_corpus -t erosionbench-sandbox .`

### Option B: local Python (3.13)
```bash
python3.13 -m venv .venv && source .venv/bin/activate   # Python 3.11+ (not macOS system 3.9)
pip install -r erosionbench/sandbox/requirements-sandbox.txt
python -m erosionbench.sandbox.server --port 8199 --corpus-dir data/shared_corpus
```

Either way, the sandbox is then live:
- **OpenAPI docs**: http://localhost:8199/docs
- **Dashboard**:    http://localhost:8199/dashboard
- **Health**:       http://localhost:8199/admin/health

The agents interact with it purely as **REST tool-calls** — it is not a shell. Key surfaces:
- **Enterprise action tools** (the attack surface — where a payload "lands"):
  `POST /v1/billing/refunds`, `/v1/sla/waivers`, `/v1/entitlements/overrides`,
  `/v1/fulfillment/ship` (CRM); `/v1/etl/sinks/{id}/execute` (data-eng);
  `/v1/admin/overrides/execute` (CI/CD); `/v1/pipelines/{id}/execute` (finance).
- **Document API** (serves the corpus PDFs): `GET /v1/docs/corpus/{domain}`,
  `GET /v1/docs/fetch/{doc_id}`, `GET /v1/docs/pdf/{doc_id}`.
- **Admin/telemetry** (how the auditor + scoring observe): `/admin/audit-log`,
  `/admin/blast-radius`, `/admin/state`, `/admin/reset`.

---

## Verify integrity

The corpus PDFs are LLM-generated and **NOT byte-reproducible from the seed**, so verifiability is
**freeze + prove**: ship the frozen bytes, and prove integrity by re-hashing.

```bash
# 1. every shipped file matches its recorded SHA-256 (42 files; paths are relative to data/):
(cd data && grep '^[0-9a-f]' PROVENANCE_hashes.txt | shasum -a 256 -c -)

# 2. the dataset composition gate (0 refusals, 0 wrapper-meta leakage):
python scripts/verify_phase1_gate.py data/dataset.json
```

`verification_manifest.json` records, per domain: each base doc's PDF-byte + page-text hash, and
one recovered-payload row per malicious scenario proving the forged key=value survived the
text→PDF→text round-trip and is capturable. All four domains, 1,728 rows, 0 malicious scenarios
missing a payload row.

---

## Provenance
- **Datagen model**: gpt-5   |   **FINAL_SEED**: 20260724001   |   **split seed**: 1337 (fraction 0.65)
- **Domains**: autonomous_crm, cicd_supply_chain, data_engineering, quantitative_finance
- **16 attack classes** × 4 domains; 81 malicious scenarios per class in the full pool.
