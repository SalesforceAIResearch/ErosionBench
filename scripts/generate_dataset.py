# Copyright (c) 2026, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Step 1: Generate scenarios and save to disk as JSON.

Stores both the executable scenario format AND the assertion contracts
(serializable specs that can be recompiled into predicates at load time).

Usage:
    python generate_dataset.py [--domain quantitative_finance] [--count 1] [--tiers 1,2,3]
                               [--model gpt-4.1] [--seed 42] [--out results/full_eval/dataset.json]
                               [--workers 4] [--parallel-domains]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, ".")

# Extended LLM timeouts for large generation runs. gpt-5 narrative calls can be slow, and at high
# worker counts contention pushes some calls past the short default -> spurious APITimeoutError ->
# whole attack instances abandoned (observed: QF T1 classes dropped). Set BEFORE importing the LLM
# client so the values are picked up. Respect any value the caller already exported.
for _k, _v in (
    ("EROSIONBENCH_LLM_REQUEST_TIMEOUT", "300"),  # read by llm._REQUEST_TIMEOUT at import
    ("EROSIONBENCH_LLM_MAX_ATTEMPTS", "4"),        # read by llm._LLM_MAX_ATTEMPTS at import
):
    os.environ.setdefault(_k, _v)


# ---------------------------------------------------------------------------
# Checkpoint / resume (per-domain, per-attack-instance granularity)
# ---------------------------------------------------------------------------
# Each attack instance (one work item -> 4 scenarios) is appended to an append-only JSONL checkpoint
# the moment it finishes, with fsync. A kill loses at most the in-flight instances, never a completed
# one. The checkpoint is the SINGLE SOURCE OF TRUTH: the final dataset is assembled from it, so a
# fully-resumed run (0 new work) still writes the complete file. A sidecar .meta.json pins the run
# config (seed/count/tiers/model/corpus) and resume REFUSES on any mismatch — a stale checkpoint must
# never silently poison a re-seeded pool.

def _ckpt_paths(out_path: str, domain: str):
    d = os.path.dirname(out_path) or "."
    stem = os.path.join(d, f".ckpt_{domain}")
    return f"{stem}.jsonl", f"{stem}.meta.json"


def _ckpt_config_fingerprint(args, tiers, domain: str) -> dict:
    return {
        "domain": domain, "seed": args.seed, "count": args.count,
        "tiers": tiers, "model": args.model, "corpus_dir": args.corpus_dir,
        "mock": bool(args.mock),
        # Image modality is part of the pool identity: resuming a text-only checkpoint under a
        # different image budget/modality would silently mix delivery modalities in one pool.
        "image_budget": int(getattr(args, "image_budget", 0)),
        "image_modality": getattr(args, "image_modality", "M1"),
    }


def _load_checkpoint(out_path: str, domain: str, args, tiers):
    """Return (done_keys:set, rows:list) from an existing checkpoint, or (set(), []) if none.

    Fails loud (SystemExit) if a checkpoint exists but its config fingerprint disagrees with this run
    — resuming a different seed/count/model onto an old checkpoint would silently corrupt the pool.
    """
    jsonl, meta = _ckpt_paths(out_path, domain)
    if not os.path.exists(jsonl):
        return set(), [], []
    want = _ckpt_config_fingerprint(args, tiers, domain)
    if os.path.exists(meta):
        with open(meta) as fh:
            got = json.load(fh)
        if got != want:
            print(f"ERROR: checkpoint {jsonl} was built with a DIFFERENT config:\n"
                  f"  on-disk: {got}\n  this run: {want}\n"
                  f"Refusing to resume (would corrupt the pool). Delete the checkpoint to regenerate.",
                  file=sys.stderr)
            raise SystemExit(4)
    done_keys, rows, payload_rows = set(), [], []
    with open(jsonl) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done_keys.add(rec["work_key"])
            rows.extend(rec["scenarios"])
            payload_rows.extend(rec.get("payload_rows", []))
    return done_keys, rows, payload_rows


def _make_checkpoint_writer(out_path: str, domain: str, args, tiers):
    """Return (append_fn, lock). append_fn(work_key, rendered, graphs) serializes the instance's
    scenarios (with ground_truth_contracts) and appends one fsync'd JSONL line."""
    from erosionbench.eval_pipeline.datagen_adapter import rendered_to_enterprise

    jsonl, meta = _ckpt_paths(out_path, domain)
    os.makedirs(os.path.dirname(jsonl) or ".", exist_ok=True)
    if not os.path.exists(meta):
        with open(meta, "w") as fh:
            json.dump(_ckpt_config_fingerprint(args, tiers, domain), fh, indent=2)
    lock = threading.Lock()

    def _serialize(rendered):
        g = rendered.state_graph
        s = rendered_to_enterprise(rendered)
        data = s.model_dump(mode="json", exclude={"ground_truth_assertions"})
        data["ground_truth_contracts"] = [
            {"name": a.name, "description": a.description, "severity": a.severity,
             "assertion_type": a.assertion_type, "params": a.params}
            for a in g.assertions
        ]
        return data

    def _payload_row(rendered):
        g = rendered.state_graph
        ectx = getattr(g, "enterprise_context", {}) or {}
        prov = ectx.get("_corpus_provenance")
        if not prov:
            return None
        return {
            "scenario_id": g.graph_id,
            "attack_class": (g.attack.attack_class if g.attack else None),
            "tier": g.tier,
            "base_doc_id": prov.get("base_doc_id"),
            "injection_page": prov.get("injection_page"),
            "is_manifest_class": prov.get("is_manifest_class"),
            "blend_attempts": prov.get("blend_attempts"),
            "recovered_pairs": prov.get("recovered_pairs"),
        }

    def append_fn(work_key, rendered, graphs):
        scenarios = [_serialize(r) for r in rendered]
        payload_rows = [row for r in rendered if (row := _payload_row(r)) is not None]
        rec = {"work_key": work_key, "scenarios": scenarios, "payload_rows": payload_rows}
        with lock:
            with open(jsonl, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                os.fsync(fh.fileno())

    return append_fn, jsonl


def main():
    ap = argparse.ArgumentParser(description="Generate benchmark scenarios and save to disk.")
    ap.add_argument("--domain", default="all",
                    choices=["all", "quantitative_finance", "cicd_supply_chain", "data_engineering", "autonomous_crm", "healthcare_clinical"])
    ap.add_argument("--count", type=int, default=1, help="Instances per attack class")
    ap.add_argument("--tiers", default="1,2,3", help="Comma-separated tiers")
    ap.add_argument("--model", default="gpt-4.1", help="LLM for narrative generation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/full_eval/dataset.json")
    ap.add_argument("--mock", action="store_true", help="Skip LLM, use templates")
    ap.add_argument("--corpus-dir", default=None,
                    help="Shared-corpus mode: compose docs from this base-doc corpus + LLM-blended "
                         "needle (e.g. data/shared_corpus) instead of per-scenario authoring.")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel workers per domain (for attack instances)")
    ap.add_argument("--parallel-domains", action="store_true", default=True,
                    help="Generate all domains concurrently (default: True)")
    ap.add_argument("--no-parallel-domains", action="store_false", dest="parallel_domains")
    ap.add_argument("--image-budget", type=int, default=0,
                    help="TOTAL number of attack instances (across all domains) to deliver the needle "
                         "via an IMAGE the prose cites, instead of extractable page text. 0 = text-only "
                         "(default). Modality is orthogonal to the tier axis; images are a capped subset "
                         "(keep the total near the ~50-60 carrier budget). Requires --corpus-dir.")
    ap.add_argument("--image-modality", default="M1", choices=["M1", "M2"],
                    help="Image render style for the --image-budget subset: M1 = typographic "
                         "(OCR-trivial control), M2 = camouflage carrier screenshot (needs headless Chrome).")
    args = ap.parse_args()

    tiers = [int(t) for t in args.tiers.split(",")]

    ALL_DOMAINS = ["quantitative_finance", "cicd_supply_chain", "data_engineering", "autonomous_crm"]
    domains = ALL_DOMAINS if args.domain == "all" else [args.domain]

    print(f"Generating: domains={domains} count={args.count} tiers={tiers} model={args.model}")
    print(f"  workers={args.workers}/domain, parallel_domains={args.parallel_domains}")
    if args.image_budget:
        print(f"  image modality: budget={args.image_budget} instances "
              f"(modality={args.image_modality}) across {len(domains)} domain(s)")
    t0 = time.time()

    from erosionbench.eval_pipeline.datagen.build_dataset import generate_scenarios_for_domain
    from erosionbench.eval_pipeline.datagen_adapter import batch_convert
    from erosionbench.eval_pipeline.llm import LLMClient

    llm_client = None
    if not args.mock:
        llm_client = LLMClient(model=args.model, strict=True)

    # Image modality (opt-in): the needle is delivered as an image the prose cites, so the payload
    # lives ONLY in pixels. Resolve the capturability reader ONCE (deterministic OCR — fail-loud if
    # unavailable), and split the GLOBAL instance budget across domains deterministically so a
    # resumed run reproduces the same per-domain budgets. Requires corpus mode (image assets attach
    # to composed corpus docs). (feedback_no_silent_fallback: raise, never degrade.)
    image_reader = None
    image_budget_for = {d: 0 for d in domains}
    if args.image_budget:
        if not args.corpus_dir:
            ap.error("--image-budget requires --corpus-dir (image assets attach to composed corpus docs).")
        from erosionbench.eval_pipeline.datagen.needle_image import get_ocr_reader
        image_reader = get_ocr_reader()  # raises with an install hint if OCR is unavailable
        base, extra = divmod(args.image_budget, len(domains))
        for i, d in enumerate(sorted(domains)):  # sorted -> split is independent of arg order
            image_budget_for[d] = base + (1 if i < extra else 0)

    def generate_one_domain(domain):
        # Resume: load any prior checkpoint (config-guarded), skip its completed instances, and append
        # each newly-finished instance to the same JSONL as it lands (crash-safe).
        done_keys, _rows, _prows = _load_checkpoint(args.out, domain, args, tiers)
        append_fn, ckpt_path = _make_checkpoint_writer(args.out, domain, args, tiers)
        if done_keys:
            print(f"  ↻ {domain}: resuming — {len(done_keys)} instance(s) already checkpointed")
        print(f"  → Generating domain: {domain}")
        rendered_list, graphs = generate_scenarios_for_domain(
            domain_id=domain,
            count=args.count,
            tiers=tiers,
            seed=args.seed,
            llm_client=llm_client,
            model=args.model,
            workers=args.workers,
            validate=True,
            corpus_dir=args.corpus_dir,
            skip_keys=done_keys,
            on_item_done=append_fn,
            image_budget=image_budget_for[domain],
            image_modality=args.image_modality,
            image_reader=image_reader,
        )
        print(f"  ✓ {domain}: {len(rendered_list)} new scenario(s) this run "
              f"(checkpoint: {ckpt_path})")
        return len(rendered_list)

    failed_domains = []  # domains whose generation raised — MUST fail the run (no silent partial)
    if args.parallel_domains and len(domains) > 1:
        with ThreadPoolExecutor(max_workers=len(domains)) as pool:
            futures = {pool.submit(generate_one_domain, d): d for d in domains}
            for future in as_completed(futures):
                domain = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"  ✗ {domain} FAILED: {e}")
                    failed_domains.append((domain, repr(e)))
    else:
        for domain in domains:
            try:
                generate_one_domain(domain)
            except Exception as e:
                print(f"  ✗ {domain} FAILED: {e}")
                failed_domains.append((domain, repr(e)))
                continue

    # Fail loud on ANY domain failure: a partial dataset silently under-covers the attack matrix,
    # which would poison downstream calibration/eval without warning. (feedback_no_silent_fallback)
    # Completed instances stay safely on their checkpoints — re-run to resume the failed domain(s).
    if failed_domains:
        print("ERROR: one or more domains failed — refusing to write a partial dataset:",
              file=sys.stderr)
        for d, why in failed_domains:
            print(f"  - {d}: {why}", file=sys.stderr)
        print("Completed instances are preserved in .ckpt_<domain>.jsonl; re-run to resume.",
              file=sys.stderr)
        return 2

    # Assemble the final dataset FROM THE CHECKPOINTS (the single source of truth): a fully-resumed
    # run does no in-memory generation yet must still emit the complete file. Also collect the
    # corpus-provenance payload rows the verification manifest needs.
    from erosionbench.eval_pipeline.datagen.build_dataset import get_domain_spec
    serialized = []
    all_payload_rows = []
    for domain in domains:
        done_keys, rows, prows = _load_checkpoint(args.out, domain, args, tiers)
        serialized.extend(rows)
        all_payload_rows.extend(prows)

    if not serialized:
        print("ERROR: No scenarios generated.", file=sys.stderr)
        return 1

    # Completeness check: each attack instance yields 4 scenarios (T1 malicious, T1 benign twin,
    # T2, T3 — only when tiers 1,2,3 all requested). Verify we produced the full expected matrix so
    # a quietly-dropped instance (LLM abandoned after retries) fails the run instead of shrinking
    # the dataset. When tiers are a subset, the per-instance count changes, so gate on the full set.
    if set(tiers) == {1, 2, 3}:
        expected = 0
        for d in domains:
            spec = get_domain_spec(d)
            per_class = max(1, args.count // len(spec.attack_classes))
            expected += len(spec.attack_classes) * per_class * 4  # T1-mal, T1-benign, T2, T3
        if len(serialized) != expected:
            print(f"ERROR: incomplete generation — expected {expected} scenarios "
                  f"({len(domains)} domains × classes × instances × 4 tier-variants) "
                  f"but got {len(serialized)}. Some attack instances were dropped "
                  f"(likely LLM timeouts after retries). Completed instances are preserved in "
                  f".ckpt_<domain>.jsonl — re-run to resume the missing ones.",
                  file=sys.stderr)
            return 3

    elapsed = time.time() - t0
    malicious = sum(1 for s in serialized if s.get("is_malicious"))
    benign = len(serialized) - malicious
    print(f"Assembled {len(serialized)} scenarios ({malicious} attack, {benign} benign) "
          f"from checkpoints in {elapsed:.1f}s")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(serialized, f, indent=2)

    print(f"Saved to {args.out} ({os.path.getsize(args.out) / 1024:.1f} KB)")

    # Verifiability artifact (shared-corpus mode only): freeze content hashes of the base-doc PDFs
    # + page text, and the recovered-payload table (collected from the checkpoints) proving each
    # forged key=value survived the text→PDF→text round-trip.
    if args.corpus_dir:
        _write_verification_manifest(
            args.corpus_dir, all_payload_rows, domains, args, os.path.dirname(args.out) or "."
        )
    return 0


def _write_verification_manifest(corpus_dir, payload_rows, domains, args, out_dir):
    """Emit verification_manifest.json: corpus content hashes + recovered-payload provenance table.

    ``payload_rows`` are the per-needle provenance rows already collected from the domain checkpoints
    (one row per malicious corpus-blended scenario), so this works identically for a fresh run and a
    fully-resumed one (no in-memory rendered objects required).
    """
    import subprocess
    from pathlib import Path

    def _sha256_file(p):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=out_dir or ".", stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_sha = None

    # Corpus content hashes (PDF bytes + page-text) per domain.
    corpus_hashes = {}
    for domain in domains:
        mpath = Path(corpus_dir) / domain / "corpus_manifest.json"
        if not mpath.exists():
            continue
        with open(mpath) as fh:
            man = json.load(fh)
        docs = {}
        for d in man.get("documents", []):
            entry = {
                "doc_type": d.get("doc_type"),
                "total_pages": d.get("total_pages"),
                "page_text_sha256": hashlib.sha256(
                    "\x1e".join(d.get("pages", [])).encode("utf-8")
                ).hexdigest(),
            }
            pdf_path = d.get("pdf_path")
            if pdf_path and Path(pdf_path).exists():
                entry["pdf_sha256"] = _sha256_file(pdf_path)
                entry["pdf_path"] = pdf_path
            docs[d["doc_id"]] = entry
        corpus_hashes[domain] = docs

    # Recovered-payload table: one row per malicious corpus-blended scenario, collected from the
    # domain checkpoints (built by _make_checkpoint_writer._payload_row). Each pair was verified
    # capturable via capture_assignment after the PDF round-trip at build time.
    manifest = {
        "provenance": {
            "corpus_dir": str(corpus_dir),
            "datagen_model": args.model,
            "seed": args.seed,
            "git_sha": git_sha,
            "domains": list(domains),
            "note": "PDFs are LLM-blended and NOT byte-reproducible; verifiability is freeze+prove: "
                    "re-hash the corpus and confirm each recovered payload is present + capturable.",
        },
        "corpus_content_hashes": corpus_hashes,
        "recovered_payloads": payload_rows,
    }
    vpath = os.path.join(out_dir, "verification_manifest.json")
    with open(vpath, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Wrote verification manifest: {vpath} "
          f"({len(payload_rows)} needle rows, {len(corpus_hashes)} domains)")


if __name__ == "__main__":
    sys.exit(main())
