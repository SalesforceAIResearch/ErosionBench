# Sample scenarios — run ErosionBench end-to-end without downloading the dataset

This directory holds a **tiny, self-contained slice** of the benchmark so a reviewer can run the
harness end-to-end in a few minutes, without fetching the full dataset + PDF corpus.

`sample_dataset.json` — **32 scenarios spanning all four domains** (24 malicious + 8 benign twins).
**Every scenario is drawn from the held-out `eval.json` split** — the sample is a clean *subset* of
the split the headline number is measured on, so it shares no family with the calibration set. (You
can verify this: the frozen calibration artifact's leak guard, `assert_not_fit_on`, accepts this
sample — see *Calibrated option* below.)

The malicious scenarios cover **all 16 attack classes** (4 per domain). Twelve classes are
**inline-representable** — their adversarial context ships whole inside `long_context_documents`. The
remaining four are **corpus-mode**: their needle is injected at run time into a served PDF corpus, so
a **small corpus slice** ships alongside in `shared_corpus/` (see *Corpus slice* below). Several
classes include a **T2 escalation** (`…_T2`) — and two inline classes a full **T1→T2→T3 triplet**
(`CS_SPLIT_EV_0082`, `QF_REASONIN_0054`) — so a run shows the escalating-conditioning axis:

| domain | malicious | benign twin(s) |
|---|---|---|
| CI/CD supply chain    | `CS_AUTHORIT_0029` (authority escalation), `CS_SUPPLY_C_0003` (supply-chain inject), `CS_SPLIT_EV_0082` (split-evidence, T1→T2→T3), `CS_COMPLIAN_0054` (+`_T2`, compliance theater — **corpus**) | `CS_AUTHORIT_0029_CONTROL`, `CS_COMPLIAN_0054_CONTROL` |
| Autonomous CRM        | `CR_WARRANTY_0083` (warranty fraud), `CR_CRISIS_E_0027` (crisis-escalation hijack), `CR_SLA_METR_0001` (+`_T2`, SLA-metric gaming — **corpus**), `CR_ENTITLEM_0054` (+`_T2`, entitlement manipulation — **corpus**) | `CR_CRISIS_E_0027_CONTROL`, `CR_SLA_METR_0001_CONTROL`, `CR_ENTITLEM_0054_CONTROL` |
| Quantitative finance  | `QF_SEMANTIC_0003` (semantic-state poisoning), `QF_TEMPORAL_0082` (temporal exploit), `QF_REASONIN_0054` (reasoning influence, T1→T2→T3), `QF_IDENTITY_0030` (identity-spoofing gaslight) | `QF_REASONIN_0054_CONTROL` |
| Data engineering      | `DE_SHARED_M_0004` (shared-memory pollution), `DE_POLICY_R_0027` (policy reframing), `DE_LINEAGE__0089` (lineage forgery), `DE_INTERMED_0056` (+`_T2`, intermediate-step leakage — **corpus**) | `DE_SHARED_M_0004_CONTROL`, `DE_INTERMED_0056_CONTROL` |

The benign twins are **true same-family controls** (`…_CONTROL`): each is the *same* enterprise task
as its malicious sibling but carrying the *authorized* value instead of the attack, so they are the
over-refusal arm (does the agent still complete the legitimate task?).

### Corpus slice

The four **corpus-mode** classes above read a served PDF corpus rather than inline text. A minimal
slice ships in [`shared_corpus/`](shared_corpus/): **12 base documents across 3 domains** (4 per
domain) — both the rendered `*.pdf` files (for inspection) and the per-domain `corpus_manifest.json`
the sandbox actually serves from. The malicious payload is **not** in these base documents — it is a
runtime-injected needle (server-side, from each scenario's `document_manifest`), exactly as in the
full benchmark. To run the corpus-mode scenarios, point the sandbox at this slice with
`--corpus-dir data/sample/shared_corpus` (see *Run it*). The inline scenarios ignore it, so it is safe
to pass the flag for every run.

This sample is for a **smoke test** — "does the harness run and produce ASR + detection tables across
all four domains and all 16 classes?" — not for reproducing the headline calibrated numbers (that
needs the *full* held-out `eval.json`, not this 32-scenario slice; see the repo `README.md` →
*Reproduce the headline number*).

## Run it

**Start the sandbox first.** The mesh executes tool calls against the sandbox *during* the run, so
it must already be listening **before** you launch either eval below. Start it once:

```bash
# The sandbox tool server needs no API keys. Point --corpus-dir at the bundled corpus slice so the
# 4 corpus-mode classes can be served; the 12 inline classes ignore it.
python3 -m erosionbench.sandbox.server --port 8200 --corpus-dir data/sample/shared_corpus &
export EROSIONBENCH_SANDBOX_URL=http://127.0.0.1:8200
```

Sandbox required? For **Option A** it is *optional for the inline classes only* — if
`EROSIONBENCH_SANDBOX_URL` is unset, `run_full_eval` falls back to an in-process simulator (lower
fidelity, no separate server), but that fallback does **not** serve the PDF corpus, so the 4
corpus-mode scenarios need a live sandbox started with `--corpus-dir` as above. For **Option B**
(`run_final_eval`) always start the sandbox with `--corpus-dir`; it expects a live server.

### Option A — uncalibrated L5 (model-agnostic smoke test)

`run_full_eval.py` runs the whole L1–L5 ladder with a **live, uncalibrated** auditor (conformal
τ=0.5, no frozen Platt scaling). It works with **any** mesh/auditor models your keys cover.

```bash
# NOTE: this run calls the mesh + auditor models, so it DOES require API keys (set them in .env).
python3 scripts/run_full_eval.py \
    --dataset data/sample/sample_dataset.json \
    --mesh-model gpt-5.5 \
    --auditor-model claude-opus-4-8 \
    --layers L1,L2,L3,L4,L5 \
    --concurrency 2 \
    --out results/sample_run
```

Swap in any models your keys cover (e.g. `--mesh-model gpt-4o --auditor-model claude-sonnet-4-6`);
the uncalibrated path is model-agnostic.

### Option B — calibrated L5 (frozen operating point)

To exercise the **calibrated** detector, use the frozen calibration artifact
[`artifacts/artifact.json`](../../artifacts/) (the same one used for the headline). `run_final_eval.py` loads it, applies its frozen
Platt scaling + conformal τ, and **leak-guards** the run: `assert_not_fit_on` refuses if any sample
family overlaps a calibration family (it passes here — the sample is held-out `eval.json`). The
artifact is **auditor-specific**, so the auditor model *must* be the one it was fit on
(`claude-opus-4-8`); a mismatch fails loud.

```bash
export EVAL_DATASET=data/sample/sample_dataset.json
export EVAL_ARTIFACT=artifacts/artifact.json       # frozen Platt + conformal τ (opus-4-8)
export EVAL_MESH_MODEL=gpt-5.5
export EVAL_AUDITOR_MODEL=claude-opus-4-8           # MUST match the artifact
export EVAL_OUT_DIR=results/sample_calibrated
python3 scripts/run_final_eval.py
```

Both options need API access for the mesh + auditor models — set your provider keys / gateway in
`.env` first (see the repo `README.md` → *Install*).

### Run only a few scenarios

Even 32 scenarios × the full L1–L5 ladder is a lot of model calls. To spot-check the pipeline on just
the first few scenarios (file order), cap the count — both entry points support it:

```bash
# Option A: first 4 scenarios only
python3 scripts/run_full_eval.py --dataset data/sample/sample_dataset.json \
    --mesh-model gpt-5.5 --auditor-model claude-opus-4-8 --layers L1,L2,L3,L4,L5 \
    --limit 4 --concurrency 2 --out results/sample_smoke

# Option B: first 4 scenarios only
EVAL_LIMIT=4 python3 scripts/run_final_eval.py   # (with the EVAL_* exports from Option B above)
```

`--limit` / `EVAL_LIMIT` is a **smoke cap, not a valid metric** — 4 scenarios estimate nothing; it is
only for confirming the run works. The first scenarios in the file are inline classes, so a small cap
does not exercise the corpus-mode path — raise the cap (or drop it) to reach the corpus scenarios.

### Trying other auditors and mesh models

**Auditor (Option B).** [`artifacts/`](../../artifacts/) ships the frozen calibration artifact for
**five** auditor models, so you can run the calibrated path with any of them — just point `EVAL_ARTIFACT` at the
matching file and set `EVAL_AUDITOR_MODEL` to the same model (the artifact is auditor-specific and
`run_final_eval` fails loud on a mismatch):

| auditor model | `EVAL_ARTIFACT` |
|---|---|
| `claude-opus-4-8` (anchor / paper headline) | `artifacts/artifact.json` |
| `claude-sonnet-4-6` | `artifacts/artifact_sonnet46.json` |
| `gpt-5.1` | `artifacts/artifact_gpt51.json` |
| `gpt-5.5` | `artifacts/artifact_gpt55.json` |
| `gemini-3.5-flash` | `artifacts/artifact_gemini35flash.json` |

For **Option A** (uncalibrated), the auditor needs no artifact — just pass any `--auditor-model` your
keys cover.

**Mesh.** L1–L4 depend only on the *mesh* model (`--mesh-model` / `EVAL_MESH_MODEL`); swap in any
model your keys cover. Note the frozen artifacts were all calibrated against a **`gpt-5.5` mesh**, so
for meaningful *calibrated* behavior (Option B) keep the mesh at `gpt-5.5`; Option A is free to vary
both roles since it is only a smoke test.

**No API keys at all?** You can still exercise the full generation → mesh → auditor → scoring loop
**fully offline** with scripted agents:

```bash
python3 scripts/run_eval.py --mock          # no keys, no gateway, no sandbox — prints ASR + P/R/F1
```

## What to expect

The **landing** attack classes (authority, supply-chain, semantic-state, shared-memory,
policy-reframing, warranty, split-evidence, reasoning-influence, and the corpus-mode compliance
theater / SLA-metric gaming / entitlement manipulation / intermediate-step leakage) reach the
malicious end-state at L2 (no defense) and get flagged by the L5 Meta-Auditor. The corpus-mode
classes only land once the sandbox is started with `--corpus-dir data/sample/shared_corpus` (without
it the served corpus is empty and the needle has nothing to attach to).

Three classes are **genuinely agent-resisted** — they land 0/N in both arms because the policy agent
categorically refuses, which is **expected, not a bug**: `crisis_escalation_hijack`,
`lineage_forgery`, and `identity_spoofing_gaslight`. They are included on purpose: a key finding of
the benchmark is that some attacks never land a tool, so *landed*-recall and *intent*-recall are
different axes (the auditor still sees the malicious intent even when nothing executes).

Exact per-run outcomes vary with the models used — this is a **functional smoke test**, not a
fixed-oracle regression, so treat the printed tables as a demonstration that the pipeline runs, not
as calibrated benchmark values.

> **Note on the benign FPR here — do not read either option's FPR as the paper's number.**
> - **Option A (uncalibrated):** at the naive τ=0.5 point the auditor over-flags benign work, so on
>   these 8 benign twins you may see most of them marked as detections. **This is
>   expected** — moving off τ=0.5 is exactly what calibration does.
> - **Option B (calibrated):** applies the frozen split-conformal operating point (recall at ~3% FPR).
>   The code path is real, but **n=8 benign twins is far too small to estimate a 3% FPR** — the
>   headline FPR is a property of the *whole* held-out split (604 scenarios), estimated over the full
>   calibration pool. Reproduce it via the repo `README.md` → *Reproduce the headline number* (same
>   `run_final_eval.py`, pointed at the full `eval.json`). On this slice it is a functional check that
>   the calibrated code path executes and the leak guard passes, not a metric.
