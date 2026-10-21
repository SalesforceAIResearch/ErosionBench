# ErosionBench — Multi-Agent Alignment Decay Benchmark

ErosionBench measures **alignment decay**: how attack success against an LLM agent system escalates
as the system moves from a single agent to a cooperating multi-agent mesh, and how much of that decay
each defense layer recovers. Each scenario is a realistic enterprise task whose documents carry a
blended adversarial "needle"; the benchmark scores whether the malicious end-state is reached (attack
success) and whether a defense flags it — while tracking over-refusal on benign twins.

> **Scope:** enterprise *agentic* tasks in 4 domains. This is **not** a general chat-jailbreak
> benchmark, and it is **not for training** an agent to perform these attacks — the payloads are
> adversarial by design. This release is for research purposes only — see
> [`AI_ETHICS.md`](AI_ETHICS.md).

📄 **Paper:** *ErosionBench: Probing Alignment Decay in Enterprise Multi-Agent Systems* (arXiv link
coming soon) · 🤗 **Dataset:** [`Salesforce/ErosionBench`](https://huggingface.co/datasets/Salesforce/ErosionBench)

- **Domains:** 4 — Autonomous CRM, CI/CD Supply Chain, Data Engineering, Quantitative Finance
- **Attack classes:** 16 (4 per domain), mapped to the OWASP LLM Top-10 (2025)
- **Tiers:** T1 / T2 / T3 — escalating adversarial *conditioning* of the same underlying contract
- **Full pool:** 1,728 scenarios (1,296 malicious, 432 benign) across 432 families
- **Released splits:** `calibration` (1,124) + `evaluation` (604) — **family-disjoint** (overlap = 0)

The dataset, dataset card, and companion corpus are released as a separate data artifact (they are
large binary files, not committed to git); this repository holds the **evaluation harness, sandbox,
and reproduction scripts**. See [`DATASET_CARD.md`](DATASET_CARD.md) for the full data description.
The full dataset + corpus (1,728 scenarios + PDF corpus) are fetched via `download_data.py` (see
*Install*); a runnable sample in [`data/sample/`](data/sample/) (see *Quickstart* below) needs no
download at all.

---

## Quickstart (5 minutes, no data download)

New here? Run the five-rung ladder on the bundled 32-scenario sample — no dataset download needed.

```bash
# 1. Install (Python 3.11+ — NOT macOS system 3.9; see Install for details)
python3.13 -m venv .venv && source .venv/bin/activate
python3 -m pip install -U pip && pip install -e .
cp .env.example .env        # add API keys for the mesh + auditor models you'll run

# 2. Start the offline sandbox tool server (needs no API keys)
python3 -m erosionbench.sandbox.server --port 8200 --corpus-dir data/sample/shared_corpus &
export EROSIONBENCH_SANDBOX_URL=http://127.0.0.1:8200

# 3. Run the whole ladder (L1–L5) on the sample
python3 scripts/run_full_eval.py \
    --dataset data/sample/sample_dataset.json \
    --mesh-model gpt-5.5 --auditor-model claude-opus-4-8 \
    --layers L1,L2,L3,L4,L5 --concurrency 2 --out results/sample_run
    # First run / short on time? Append `--limit 8` to cap it to the first 8
    # scenarios (a quick smoke pass — a cap for speed, not a valid metric).
```

You'll see attack-success per rung across the four domains: it should **climb from L1 to L2**
(alignment decay as agents start cooperating) and **fall back as L3→L5** add defenses. See
*The defense ladder* below for what each rung means.

> This is a **smoke test**, not the reported number — the headline recall/FPR needs the full held-out
> split and the calibrated detector (see *Reproduce the headline*). **No API keys at all?** Run the
> whole loop offline with scripted agents: `python3 scripts/run_eval.py --mock`.

---

## The defense ladder

Every scenario is run through five configurations ("rungs"). Each rung keeps the previous one and adds
one more defense, so the numbers read as a story: attack-success *rises* at L2 (the decay), then each
higher rung shows how much a given defense *claws back*. L1–L4 depend only on the **mesh** model;
only **L5** depends on the **auditor** model.

| Rung | What it adds | What it measures |
|---|---|---|
| **L1** | A single agent, working alone (no mesh) | Baseline — can one isolated agent be driven to the malicious action at all? |
| **L2** | The full multi-agent mesh, **no defenses** | The attack surface — how far attack-success *rises* once agents cooperate. This gap is the **alignment-decay signal**. |
| **L3** | **+ LlamaFirewall** (content/prompt perimeter) | How much a perimeter guard that inspects content recovers. |
| **L4** | **+ Progent** (least-privilege control) | How much enforcing per-agent privilege limits recovers. |
| **L5** | **+ Meta-Auditor** (calibrated semantic detector) | How much a semantic auditor recovers at a **calibrated** operating point — the headline defense. |

**Reference run anchors:** mesh = `gpt-5.5`, auditor = `claude-opus-4-8`.

### Running one or more rungs

`run_full_eval.py --layers` takes any comma-separated subset, so you can run the whole ladder or a
single rung:

```bash
--layers L1,L2,L3,L4,L5   # the full ladder (the Quickstart command)
--layers L2               # just the no-defense mesh baseline (the attack surface)
--layers L1,L2            # just the decay gap (isolation vs. cooperating mesh)
--layers L5               # just the auditor rung
```

L1–L4 need only `--mesh-model`; add `--auditor-model` whenever L5 is in the list.

---

## Install

Requires **Python 3.11+** (reference environment: 3.13). Note that macOS ships `python3` as 3.9 — use
an explicit 3.11+ interpreter to create the venv (e.g. `python3.13`, or `python3.11`/`python3.12`,
installed via Homebrew or pyenv). Check with `python3 --version` first.

```bash
git clone https://github.com/SalesforceAIResearch/ErosionBench.git && cd ErosionBench
python3.13 -m venv .venv && source .venv/bin/activate   # any 3.11+ interpreter; NOT macOS system 3.9
python3 -m pip install -U pip setuptools wheel   # editable install needs pip >= 21.3 (PEP 660)
pip install -e .                 # core: the `erosionbench` package + eval/sandbox deps
pip install -e ".[dev]"          # + pytest & pytest-asyncio (needed to run the test suite)
pip install -e ".[guards]"       # optional: real LlamaFirewall/PromptGuard for L3/L4 (see note)
cp .env.example .env             # add API keys for the mesh/auditor providers you evaluate
```

> **Optional extras.** Core install runs the full pipeline, the sample, and the sandbox. The
> `[guards]` extra pulls the real Meta **LlamaFirewall** + PromptGuard model for L3/L4; **without it
> the harness does not crash** — the perimeter layer transparently degrades to an LLM-judge/regex
> stand-in, so `--layers …L3…` still runs (at lower fidelity). Install `[guards]` to reproduce the
> paper's firewall baseline exactly. The `[dev]` extra is only needed to run `pytest`.

### Getting the data

The full dataset + corpus are **not** committed to git (large binary files) — they are hosted on the
Hugging Face Hub. Fetch everything with:

```bash
python3 scripts/download_data.py   # pulls the dataset + PDF corpus from the Hub
```

(Set `HF_TOKEN` in `.env` only if the data repo is gated — see [`.env.example`](.env.example); override
the repo id with `EROSIONBENCH_HF_REPO`.) Prefer a zero-download first look? A runnable 32-scenario
slice ships in [`data/sample/`](data/sample/) and runs the whole harness with **no download at all**
(see *Quickstart* above).

The sandbox is offline and needs **no** API keys. See [`RELEASE_README.md`](RELEASE_README.md) and
[`Dockerfile.sandbox`](Dockerfile.sandbox) to run it in a container.

---

## What's in the sample

The [`data/sample/`](data/sample/) slice the Quickstart runs on is **32 scenarios across all four
domains** (24 malicious + 8 benign twins) covering **all 16 attack classes**, every one drawn from the
**held-out `eval.json` split**. Twelve classes are *inline* (the context ships whole in the scenario);
the four *corpus-mode* classes read the small PDF slice in
[`data/sample/shared_corpus/`](data/sample/shared_corpus/) (12 base docs, needle injected at run time —
the base docs carry no payload, which is why the sandbox needs `--corpus-dir`). Several classes include
a **T2** escalation and two a full **T1→T2→T3** triplet so the escalation axis is visible; two classes
(`lineage_forgery`, `identity_spoofing_gaslight`) are agent-resisted zero-landers *by design*. See
[`data/sample/README.md`](data/sample/README.md).

- **Fewer scenarios:** add `--limit N` to the Quickstart command (a smoke cap, not a valid metric).
- **No API keys at all:** `python3 scripts/run_eval.py --mock` runs the whole generation → mesh →
  auditor → scoring loop offline with scripted agents (no keys, no sandbox).
- **A different auditor on the sample:** [`artifacts/`](artifacts/) ships frozen calibration artifacts
  for five auditors, so the *calibrated* path runs on the sample too — see *Run a different auditor* below.

---

## Two kinds of result (why there are two scripts)

There are two eval entry points, and they answer different questions — this is the one distinction
worth internalizing before you read a number off either:

| Dimension | `run_full_eval.py` | `run_final_eval.py` |
|---|---|---|
| **Answers** | How does attack-success change across the ladder? | What is the detector's recall / false-positive rate? |
| **L5 threshold** | live default (τ=0.5), **uncalibrated** | **calibrated** (frozen Platt + split-conformal τ) |
| **Split** | any dataset (sample or full) | **held-out `eval.json` only** |
| **Use it for** | the decay story + quick smoke checks | the **reported headline number** |

Start with `run_full_eval.py` to see the ladder (that's the Quickstart); use `run_final_eval.py`
below for the calibrated headline.

---

## Reproduce the headline number

> ⚠ **Requires the fetched dataset.** This path reads `data/eval_clean_control.json`, `data/shared_corpus/`, and the
> frozen `artifacts/artifact.json` — the data is obtained via `download_data.py` (see *Install*). No download yet? Use the
> bundled `data/sample/` (see *Run a different auditor → Option 1*) instead — the calibrated code path
> there is identical, only the split size (and therefore the reported numbers) differs.

The headline (L5 recall/FPR at the split-conformal operating point) is scored **once** over the
held-out **clean-control arm** (`eval_clean_control.json`: `eval.json`'s 453 malicious scenarios + 453
validity-matched benign controls) with the **frozen calibration artifact** via `run_final_eval.py`.
It is parameterized by environment variables (the values below reproduce the reference opus-4-8 run;
the script's default dataset is `data/eval.json`, the original 604-scenario split):

```bash
# 1. Start the offline sandbox tool server (serves the real PDF corpus)
python3 -m erosionbench.sandbox.server --port 8223 --corpus-dir data/shared_corpus &

# 2. Score the held-out split with the frozen artifact (mesh=gpt-5.5, auditor=claude-opus-4-8)
export EROSIONBENCH_SANDBOX_URL=http://127.0.0.1:8223
export EVAL_DATASET=data/eval_clean_control.json  # held-out clean-control arm (paper headline)
export EVAL_ARTIFACT=artifacts/artifact.json  # frozen Platt + conformal τ (auditor-specific)
export EVAL_MESH_MODEL=gpt-5.5
export EVAL_AUDITOR_MODEL=claude-opus-4-8    # MUST match the auditor the artifact was fit on
export EVAL_OUT_DIR=results/repro
python3 scripts/run_final_eval.py
```

The artifact is **auditor-specific** and **leak-guarded**: `from_artifact` refuses to run if any
evaluation family overlaps a calibration family, so a mismatched auditor or a calib/eval leak fails
loud. Report the number on the **held-out evaluation families only** (`eval_clean_control.json` or
`eval.json`); the calibration split is for
fitting the detector's operating point. Scoring is **fail-closed**: scenarios that error out are
retried and then **excluded** from the denominator (checkpointed to JSONL, so a crash resumes), so
the effective *n* can be below the nominal split size — report it.

---

## Run a different auditor (calibrated)

L5 is the only auditor-dependent layer, and its frozen artifact bundles the Platt scaling +
split-conformal threshold **fit for one specific auditor against a `gpt-5.5` mesh**. To evaluate a
*different* auditor at a calibrated operating point you have two paths — and the first one **runs on
the bundled sample with no data download**, so you can try a different auditor right now.

### Option 1 — reuse a pre-fit artifact (five auditors ship ready) — *runs on the sample*

The [`artifacts/`](artifacts/) directory ships frozen calibration artifacts for **five** auditors, so
with the bundled [`data/sample/`](data/sample/) this is the **same calibrated code path as the headline,
just on the 32-scenario sample**. Point
`EVAL_ARTIFACT` and `EVAL_AUDITOR_MODEL` at a **matching** pair (`run_final_eval` fails loud on a
mismatch):

| auditor model | `EVAL_ARTIFACT` |
|---|---|
| `claude-opus-4-8` (anchor / paper headline) | `artifacts/artifact.json` |
| `claude-sonnet-4-6` | `artifacts/artifact_sonnet46.json` |
| `gpt-5.1` | `artifacts/artifact_gpt51.json` |
| `gpt-5.5` | `artifacts/artifact_gpt55.json` |
| `gemini-3.5-flash` | `artifacts/artifact_gemini35flash.json` |

```bash
# Start the sandbox first (needs no keys). --corpus-dir serves the bundled corpus slice so the
# corpus-mode classes run; inline classes ignore it.
python3 -m erosionbench.sandbox.server --port 8223 --corpus-dir data/sample/shared_corpus &
export EROSIONBENCH_SANDBOX_URL=http://127.0.0.1:8223

export EVAL_DATASET=data/sample/sample_dataset.json      # bundled sample — no download needed
export EVAL_ARTIFACT=artifacts/artifact_sonnet46.json    # matching artifact for the auditor below
export EVAL_MESH_MODEL=gpt-5.5                            # keep gpt-5.5: the artifacts were calibrated against it
export EVAL_AUDITOR_MODEL=claude-sonnet-4-6              # MUST match the artifact
export EVAL_OUT_DIR=results/sample_sonnet46
python3 scripts/run_final_eval.py                        # needs API keys for the mesh + auditor
```

Keep `EVAL_MESH_MODEL=gpt-5.5` — the calibration (and hence the operating point) is joint over the
auditor **and** the mesh, so swapping the mesh invalidates the frozen threshold. Each artifact was fit
on the full held-out calibration split, so the **same command reproduces that auditor's headline** once
you point `EVAL_DATASET` at the full `data/eval.json` (fetched data — see *Install*); on the
32-scenario sample it is a functional check that the calibrated path runs and the leak guard passes,
not a metric (n is far too small to estimate the operating point).

### Option 2 — fit a brand-new auditor

> ⚠ **Requires the fetched dataset.** Fitting reads the calibration split `data/calib.json`, which is
> **not** in the code bundle (fetch it via `download_data.py` — see *Install*). To try a *new* auditor
> without the download, fit on the bundled sample instead
> (`LIVE_DATASET=data/sample/sample_dataset.json`), but a 32-scenario slice is far too small for a
> trustworthy operating point (`collect_calibration.py` will print `sufficient: False`) — it only
> exercises the fitting code path.

For an auditor that has no pre-fit artifact, fit one on the **calibration split** with
`scripts/collect_calibration.py` (it runs the mesh + auditor over the calibration families and freezes
the Platt + conformal artifact), then score `eval.json` with it. Fitting needs the calibration split, a
running sandbox, and API access for the mesh + auditor.

```bash
# 1. Fit the operating point for the new auditor on the calibration split (mesh stays gpt-5.5)
export EROSIONBENCH_SANDBOX_URL=http://127.0.0.1:8223
export LIVE_DATASET=data/calib.json          # calibration split — becomes the artifact's calib families
export LIVE_MESH_MODEL=gpt-5.5
export LIVE_AUDITOR_MODEL=claude-haiku-4-5-20251001   # the auditor you want to calibrate
export TARGET_FPR=0.05                        # target benign false-positive rate for the conformal τ
export CAL_OUT=results/calib/artifact_haiku45.json
python3 scripts/collect_calibration.py        # writes CAL_OUT (+ a _records.json alongside)

# 2. Score the held-out split with the artifact you just fit
export EVAL_DATASET=data/eval.json
export EVAL_ARTIFACT=results/calib/artifact_haiku45.json
export EVAL_MESH_MODEL=gpt-5.5
export EVAL_AUDITOR_MODEL=claude-haiku-4-5-20251001    # MUST match what you fit in step 1
export EVAL_OUT_DIR=results/repro_haiku45
python3 scripts/run_final_eval.py
```

`collect_calibration.py` fits **only** on the calibration split, so the families it touches become the
artifact's calibration families; the same leak guard then refuses to score any evaluation family that
overlaps them. Check the printed `sufficient: True` — if the benign/family counts are too low the
conformal threshold is not trustworthy.

---

## Repository structure

```
ErosionBench/
├── README.md               # this file
├── DATASET_CARD.md         # full dataset description (dataset card)
├── RELEASE_README.md       # sandbox / release packaging
├── pyproject.toml          # pip install -e .
├── Dockerfile.sandbox
├── LICENSE.txt             # Apache-2.0 (code); dataset/corpus are CC-BY-4.0 (DATA_LICENSE)
├── DATA_LICENSE
├── CITATION.cff
├── AI_ETHICS.md · SECURITY.md · CONTRIBUTING.md · CODE_OF_CONDUCT.md
│
├── erosionbench/           # the package
│   ├── eval_pipeline/      # mesh, meta-auditor, calibration, datagen, scoring
│   └── sandbox/            # offline FastAPI tool server + domain handlers
│
├── scripts/                # user-facing entry points
│   ├── run_final_eval.py   # headline repro: score eval.json once w/ the frozen artifact
│   ├── run_full_eval.py    # full L1–L5 ladder over a dataset (live auditor pass)
│   ├── generate_dataset.py # regenerate the scenario pool
│   ├── collect_calibration.py # fit the split-conformal detector per auditor
│   ├── download_data.py    # fetch dataset + corpus from the Hugging Face release
│   └── …                   # dataset build, Mode B (image) scoring, integrity tools
│
├── artifacts/              # frozen calibration artifacts (Platt + conformal τ), one per auditor
├── data/                   # dataset JSONs + shared_corpus/ fetched from the Hub (git-ignored);
│   └── sample/             #   bundled 32-scenario slice + 12-doc corpus (tracked, no download)
├── docs/                   # architecture + dataset-pipeline design notes
└── tests/                  # unit + integration tests (pytest)
```

---

## Integrity ("freeze + prove")

> ⚠ **Requires the fetched dataset.** The `verify_phase1_gate.py` check below reads `data/dataset.json`,
> which is **not** in the code bundle (fetch it via `download_data.py` — see *Install*). The hash check
> runs against files that ship. With only `data/sample/`, skip the gate command.

The malicious payloads are needles blended into a PDF corpus that is **not byte-reproducible from a
seed** — the PDFs are frozen, required binary artifacts. Provenance is proven by re-hashing + a
recovered-payload capturability table.

```bash
(cd data && grep '^[0-9a-f]' PROVENANCE_hashes.txt | shasum -a 256 -c -)   # all shipped files
python3 scripts/verify_phase1_gate.py data/dataset.json        # 0 refusals, 0 wrapper-meta leakage
```

Pinned `dataset.json` SHA-256 (v1.1): `faf765c73a071d70305b11e49caaca9317fc88398e89a7891039e9218bc0e3bf`.
See the [changelog](DATASET_CARD.md#changelog) for the v1.0 → v1.1 correction.

---

## Testing

```bash
pip install -e ".[dev]"  # pytest + pytest-asyncio (asyncio_mode=auto); skip if already installed
pytest tests/            # unit + integration; the sandbox tests spin up an in-process server
```

---

## Ethics, security & contributing

- **Ethics:** this release is for research purposes only in support of an academic paper — read
  [`AI_ETHICS.md`](AI_ETHICS.md) before use. The scenarios contain adversarial payloads by design;
  do not use them to train agents to perform these attacks.
- **Security:** report vulnerabilities per [`SECURITY.md`](SECURITY.md), not via public issues.
- **Contributing:** see [`CONTRIBUTING.md`](CONTRIBUTING.md) and the
  [Code of Conduct](CODE_OF_CONDUCT.md).

---

## License & citation

Dual-licensed: the **code** (the `erosionbench` package, sandbox, and scripts) is under
**Apache-2.0** ([`LICENSE.txt`](LICENSE.txt)); the **dataset and corpus** are under **CC-BY-4.0**
([`DATA_LICENSE`](DATA_LICENSE)).

```bibtex
@misc{erosionbench2026,
  title         = {ErosionBench: Probing Alignment Decay in Enterprise Multi-Agent Systems},
  author        = {Bharathwaj, Riya and Prabhakar, Akshara and Ram, Roshan and Kothari, Avni and Tan, Sarah and Wang, Huan and Savarese, Silvio},
  year          = {2026},
  eprint        = {XXXX.XXXXX},    % TODO: fill arXiv id once v1 is posted
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR},
  note          = {Generator seed 20260724001; family-disjoint calibration/evaluation splits.}
}
```
