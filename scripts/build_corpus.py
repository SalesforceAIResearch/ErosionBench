# Copyright (c) 2026, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Build the shared benign document corpus to disk (PDFs + corpus_manifest.json).

This is called ONCE per corpus, not per scenario. It writes, for each domain:
    <out>/<domain>/corpus_manifest.json      (the reloadable corpus metadata + pages)
    <out>/<domain>/<doc_id>.pdf              (rendered PDF artifact, best-effort)

The layout matches what DocumentServer.load_corpus(domain) and the sandbox
DocumentStore expect: point them at <out> and they will find <out>/<domain>/corpus_manifest.json.

Usage:
    python build_corpus.py --domain all --out dataset/corpus/ [--mock] [--model gpt-4.1] [--seed 42]

Then serve it from the sandbox:
    export SANDBOX_CORPUS_DIR=$(pwd)/dataset/corpus
    bash erosionbench/sandbox/launcher.sh --port 8199
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")

ALL_DOMAINS = ["quantitative_finance", "cicd_supply_chain", "data_engineering", "autonomous_crm"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the shared benign corpus to disk.")
    ap.add_argument("--domain", default="all", choices=["all", *ALL_DOMAINS])
    ap.add_argument("--out", default="dataset/corpus", help="Base corpus directory")
    ap.add_argument("--org-name", default="Meridian Capital Partners")
    ap.add_argument("--model", default="gpt-4.1", help="LLM for realistic page generation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mock", action="store_true", help="Skip LLM, use deterministic templates")
    args = ap.parse_args()

    from erosionbench.eval_pipeline.datagen.corpus_builder import build_domain_corpus

    llm_client = None
    if not args.mock:
        from erosionbench.eval_pipeline.llm import LLMClient
        llm_client = LLMClient(model=args.model, strict=True)

    domains = ALL_DOMAINS if args.domain == "all" else [args.domain]
    base = Path(args.out)

    total_docs = 0
    for domain in domains:
        domain_dir = base / domain
        print(f"  → Building corpus for {domain} → {domain_dir}")
        corpus = build_domain_corpus(
            domain=domain,
            output_dir=domain_dir,
            org_name=args.org_name,
            llm_client=llm_client,
            model=args.model,
            seed=args.seed,
        )
        manifest = domain_dir / "corpus_manifest.json"
        pdfs = sum(1 for d in corpus if d.pdf_path)
        print(f"  ✓ {domain}: {len(corpus)} docs ({pdfs} PDFs) → {manifest}")
        total_docs += len(corpus)

    if total_docs == 0:
        print("ERROR: No corpus documents built (no templates for the selected domain?).")
        return 1

    print(f"Built {total_docs} corpus documents across {len(domains)} domain(s) under {base}/")
    print(f"Serve it:  export SANDBOX_CORPUS_DIR={base.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
