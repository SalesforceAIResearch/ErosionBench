"""Procedural dataset generation pipeline for the ErosionBench blue team benchmark.

This package implements a 4-stage pipeline that generates adversarial scenarios
at scale across 4 enterprise domains, validates them through live multi-agent
debate, and exports to HuggingFace Parquet format.

Stages:
    1. state_graph    - Structural contracts (what happens, not how it reads)
    2. generator      - LLM narrative coherence engine (Claude Sonnet renders artifacts)
    3. mutator        - Tier intensification + benign twin pairing
    4. validator      - Live 6-baseline execution + ground-truth verification + export

Document architecture (scalable):
    - corpus_builder  - Shared enterprise document corpus (generated once per domain)
    - document_server - Runtime needle injection (per-scenario, no new PDFs)
    - pdf_renderer    - One-time PDF rendering for the fixed corpus
"""
