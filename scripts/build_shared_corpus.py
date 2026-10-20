# Copyright (c) 2026, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Build the shared base-document corpus (32 docs = 8 doc_types x 4 domains), ONCE.

Every doc is LLM-written (gpt-5) for a uniformly rich substrate the needle can blend into
(EROSIONBENCH_CORPUS_FORCE_LLM=1). Docs are long (15-50 pages) so downstream needle assembly can
slice coherent T1 (short prefix) / T2 (mid) / T3 (full) tiers with tier-appropriate injection depth.

Fail-loud: any domain that errors or returns 0 docs aborts the run (no partial corpus).

Usage:
    EROSIONBENCH_CORPUS_FORCE_LLM=1 python build_shared_corpus.py \
        --model gpt-5 --org "Meridian Capital" --seed 20260722 \
        --out results/shared_corpus
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, ".")

# Extended LLM timeouts: gpt-5 prose batches are slow; set BEFORE importing the client.
for _k, _v in (("EROSIONBENCH_LLM_REQUEST_TIMEOUT", "300"), ("EROSIONBENCH_LLM_MAX_ATTEMPTS", "4")):
    os.environ.setdefault(_k, _v)


def main():
    ap = argparse.ArgumentParser(description="Build the shared base-document corpus (32 docs).")
    ap.add_argument("--model", default="gpt-5")
    ap.add_argument("--org", default="Meridian Capital")
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--out", default="results/shared_corpus")
    args = ap.parse_args()

    # Force uniform LLM authoring across the whole pool (see corpus_content.should_use_llm).
    os.environ["EROSIONBENCH_CORPUS_FORCE_LLM"] = "1"

    from erosionbench.eval_pipeline.datagen.corpus_builder import CorpusBuilder, _CORPUS_TEMPLATES
    from erosionbench.eval_pipeline.llm import LLMClient

    domains = list(_CORPUS_TEMPLATES.keys())
    expected = {d: len(_CORPUS_TEMPLATES[d]) for d in domains}
    total_expected = sum(expected.values())
    print(f"Building shared corpus: {len(domains)} domains, {total_expected} docs, model={args.model}")
    print(f"  force_llm=1  org={args.org!r}  seed={args.seed}  out={args.out}")
    t0 = time.time()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    def build_one(domain):
        # Each thread gets its own client (LLMClient.chat is blocking HTTP).
        client = LLMClient(model=args.model, strict=True)
        builder = CorpusBuilder(llm_client=client, model=args.model)
        corpus = builder.build_corpus(
            domain=domain, org_name=args.org, output_dir=out_root / domain, seed=args.seed,
        )
        return domain, corpus

    built = {}
    failed = []
    with ThreadPoolExecutor(max_workers=len(domains)) as pool:
        futs = {pool.submit(build_one, d): d for d in domains}
        for fut in as_completed(futs):
            d = futs[fut]
            try:
                domain, corpus = fut.result()
                if not corpus:
                    failed.append((d, "0 docs"))
                else:
                    built[domain] = corpus
                    npages = sum(len(doc.pages) for doc in corpus)
                    print(f"  ✓ {domain}: {len(corpus)} docs, {npages} pages total")
            except Exception as e:
                print(f"  ✗ {d} FAILED: {e!r}", file=sys.stderr)
                failed.append((d, repr(e)))

    if failed:
        print("ERROR: corpus build failed for one or more domains — no partial corpus written:",
              file=sys.stderr)
        for d, why in failed:
            print(f"  - {d}: {why}", file=sys.stderr)
        return 2

    # Completeness gate: every expected doc must exist with non-empty pages.
    problems = []
    for d in domains:
        got = {doc.doc_type: doc for doc in built.get(d, [])}
        if len(got) != expected[d]:
            problems.append(f"{d}: got {len(got)} docs, expected {expected[d]}")
        for dt, doc in got.items():
            if not doc.pages or not any((p or "").strip() for p in doc.pages):
                problems.append(f"{d}/{dt}: empty pages")
    if problems:
        print("ERROR: incomplete corpus:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 3

    # Summary index (human-readable) alongside the per-domain corpus_manifest.json files.
    index = {
        "model": args.model, "org": args.org, "seed": args.seed,
        "force_llm": True, "domains": {},
    }
    for d in domains:
        index["domains"][d] = [
            {"doc_id": doc.doc_id, "doc_type": doc.doc_type, "title": doc.title,
             "pages": len(doc.pages),
             "chars": sum(len(p) for p in doc.pages)}
            for doc in built[d]
        ]
    with open(out_root / "corpus_index.json", "w") as f:
        json.dump(index, f, indent=2)

    elapsed = time.time() - t0
    total_docs = sum(len(built[d]) for d in domains)
    print(f"\nBuilt {total_docs} docs across {len(domains)} domains in {elapsed:.0f}s")
    print(f"Index: {out_root / 'corpus_index.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
