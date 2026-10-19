"""ErosionBench: a multi-agent alignment-decay benchmark.

This package implements the "Cross-State Memory Poisoning" threat scenario as a
self-contained Proof of Concept built on LangGraph + LangChain:

- Module 1 (``scenario_generator``): a target-varying scenario generator that
  emits an *interaction matrix* of poisoning scenarios, each corrupting a
  DIFFERENT architectural component (node / edge / state variable).
- Module 2 (``pipeline_graph``): a stateful LangGraph workflow for an
  enterprise deployment pipeline, plus a dynamic ``Gatekeeper`` defense node.
- Module 3 (``run_benchmark``): an execution harness that runs three control
  experiments (Safety Floor, Exposure Ceiling, Defended Multi-Agent).
"""

__all__ = ["scenario_generator", "pipeline_graph", "run_benchmark", "llm"]
