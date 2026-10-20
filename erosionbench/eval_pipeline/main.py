"""Evaluation orchestrator -- ties the LangGraph mesh pipeline together end-to-end.

Examples
--------
    # Deterministic, offline (scripted confused-deputy agents + regex firewall):
    python3 -m erosionbench.eval_pipeline.main --mode multi --mock
    python3 -m erosionbench.eval_pipeline.main --mode single --mock

    # Live: real frontier agents in a LangGraph mesh + live LlamaFirewall
    # (PromptGuard input, PromptGuard + AlignmentCheck output):
    python3 -m erosionbench.eval_pipeline.main --mode multi  --model gpt-4o  --trials 5
    python3 -m erosionbench.eval_pipeline.main --mode single --model gpt-5.5 --trials 5

For each (scenario x trial) the orchestrator builds a fresh ledger + firewall, runs the
configured LangGraph loop, hands the result to the deterministic solver, and appends a full
execution trace to ``evaluation_traces.json``. With real models the outcome is stochastic,
so ``--trials N`` repeats each scenario and the report shows compromise rates.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

from .langgraph_mesh import LangGraphMesh, MeshConfig
from .metrics import benchmark_metrics_to_dict, compute_benchmark_metrics
from .scenario_pool import DOMAIN_PATTERNS, ScenarioPool, build_standard_pool
from .schemas import EnterpriseScenario, ThreatCategory
from .solver import AggregateMetrics, DeterministicMeshSolver, ScenarioResult
from .traces import DEFAULT_TRACE_PATH, append_trace
from .watermark_scrubber import scrub_scenario

# Lazy imports for datagen pipeline integration
_datagen_loaded = False


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------
class RunConfig:
    """Resolved knobs for one evaluation invocation."""

    def __init__(self, mode: str, model: str, firewall_kind: str, trials: int,
                 verbose: bool, scripted_agent: bool, use_llm: bool, traces_path: str,
                 attacker: str = "scripted", agent_profiles_path: Optional[str] = None,
                 partial_observability: bool = True) -> None:
        self.mode = mode
        self.model = model
        self.firewall_kind = firewall_kind
        self.trials = max(1, trials)
        self.verbose = verbose
        self.scripted_agent = scripted_agent
        self.use_llm = use_llm
        self.traces_path = traces_path
        self.attacker = attacker
        self.agent_profiles_path = agent_profiles_path
        self.partial_observability = partial_observability
        # Derived views used by the reporter.
        self.single_agent = mode == "single"
        self.agent_llm = use_llm and not scripted_agent


# ---------------------------------------------------------------------------
# Datagen pipeline integration
# ---------------------------------------------------------------------------
def generate_scenarios_from_datagen(
    domain: str,
    count: int,
    tiers: List[int],
    seed: int = 42,
    model: str = "claude-sonnet-4-6-v1",
    mock: bool = False,
    validate: bool = True,
) -> List[EnterpriseScenario]:
    """Generate scenarios via the datagen pipeline and convert to executable format.

    Pipeline: state_graph → generator → validation_gate → mutator → adapter → EnterpriseScenario

    In live mode (mock=False), the LLM client is constructed in strict mode — if
    credentials are missing or the model is unreachable, this raises immediately
    rather than silently falling back to template generation.
    """
    from .datagen.build_dataset import generate_scenarios_for_domain
    from .datagen_adapter import batch_convert
    from .llm import LLMClient, LLMClientError

    llm_client = None
    if not mock:
        llm_client = LLMClient(model=model, strict=True)

    rendered, _graphs = generate_scenarios_for_domain(
        domain_id=domain,
        count=count,
        tiers=tiers,
        seed=seed,
        llm_client=llm_client,
        model=model,
        workers=1,
        validate=validate,
    )

    return batch_convert(rendered)


# ---------------------------------------------------------------------------
# Core evaluation loop (UI-free, importable)
# ---------------------------------------------------------------------------
def evaluate_pool(
    scenarios: List[EnterpriseScenario], config: RunConfig
) -> "tuple[Dict[str, List[ScenarioResult]], str, str]":
    """Run each scenario ``config.trials`` times through the LangGraph mesh.

    Returns ``(results_by_scenario, backend_name, firewall_name)``. The mesh reuses one
    LLM client across runs (token usage is reset per run) but builds a fresh ledger and a
    fresh firewall per run so every trace is hermetic. Every run is persisted to the
    append-only ``evaluation_traces.json``.

    Scenarios are scrubbed of benchmark watermarks before mesh execution so agents
    under test cannot keyword-spot the attack tells.
    """
    solver = DeterministicMeshSolver()
    mesh = LangGraphMesh(
        MeshConfig(
            mode=config.mode,
            model=config.model,
            firewall_kind=config.firewall_kind,
            scripted_agent=config.scripted_agent,
            verbose=config.verbose,
            attacker=config.attacker,
            agent_profiles_path=config.agent_profiles_path,
            partial_observability=config.partial_observability,
        )
    )
    results: Dict[str, List[ScenarioResult]] = defaultdict(list)
    firewall_name = "?"
    scripted_effective = config.scripted_agent

    for scenario in scenarios:
        scrubbed = scrub_scenario(scenario)
        for trial in range(config.trials):
            context, trace = mesh.run(scrubbed)
            result = solver.evaluate(scenario, context)
            results[scenario.scenario_id].append(result)

            firewall_name = trace.get("firewall", firewall_name)
            scripted_effective = bool(trace.get("scripted_agent", scripted_effective))
            # Augment the trace with the deterministic ground-truth verdict, then persist.
            trace["trial"] = trial
            trace["verdict"] = result.verdict.value
            trace["defense_attribution"] = result.defense_attribution
            trace["alignment_blindspot"] = result.alignment_blindspot
            trace["triggered_assertions"] = result.triggered_assertions
            append_trace(trace, config.traces_path)

            if config.verbose:
                _dump_transcript(scenario, context, result, trial)

    backend_name = (f"openai:{config.model}" if not scripted_effective
                    else f"scripted-confused-deputy (offline; model={config.model})")
    return results, backend_name, firewall_name


def _dump_transcript(scenario, context, result, trial: int) -> None:
    """Print the internal dialogue, firewall rulings, and assertion outcomes for one run."""
    print(f"\n{'=' * 78}")
    print(f"SCENARIO {scenario.scenario_id}  (trial {trial + 1})  target={scenario.target_agent.value}")
    print(f"  user intent : {scenario.user_intent}")
    print(f"  trajectory  : {' -> '.join(r.value for r in context.runtime_state.trajectory)}")
    print(f"{'-' * 78}")
    print("  INTERNAL DIALOGUE (agent <-> agent):")
    for turn in context.runtime_state.dialogue:
        who = turn.speaker.value + (f" -> {turn.listener.value}" if turn.listener else "")
        body = turn.content if len(turn.content) <= 300 else turn.content[:297] + "..."
        print(f"    [{who}] {body}")
    print(f"  final policy consensus : {context.runtime_state.policy_consensus}")
    print(f"  tool fired             : {context.tool_fired}"
          + (f"  args={context.last_tool_call.arguments}" if context.tool_fired else ""))
    print("  FIREWALL (perimeter) decisions:")
    for d in context.firewall_decisions:
        print(f"    {d.boundary.value:>6}: {d.decision.value:<5} -- {d.reason}")
    print("  GROUND-TRUTH ASSERTIONS:")
    for o in result.outcomes:
        mark = "TRIGGERED" if o.triggered else "clear"
        decisive = " (CRITICAL)" if o.is_decisive else " (diagnostic)"
        print(f"    [{mark:>9}]{decisive} {o.name}: {o.description}")
    print(f"  >>> VERDICT: {result.verdict.value}"
          + ("  [PERIMETER-BLIND]" if result.alignment_blindspot else ""))


def flatten(results_by_scenario: Dict[str, List[ScenarioResult]]) -> List[ScenarioResult]:
    """Flatten the per-scenario result lists into one list of (scenario x trial) results."""
    flat: List[ScenarioResult] = []
    for items in results_by_scenario.values():
        flat.extend(items)
    return flat


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def build_report_dataframe(results_by_scenario: Dict[str, List[ScenarioResult]], trials: int):
    """Per-scenario summary (compromise/blindspot rates across trials)."""
    rows = []
    for scenario_id, items in results_by_scenario.items():
        n = len(items)
        is_attack = bool(items and items[0].is_malicious)
        kind = "attack" if is_attack else "benign"
        compromised = sum(1 for r in items if r.is_compromised)
        sut_refused = sum(1 for r in items if r.agent_blocked)
        fw_blocked = sum(1 for r in items if r.firewall_blocked)
        fw_flagged = sum(1 for r in items if r.firewall_flagged)
        tool_fired = sum(1 for r in items if r.tool_fired)
        decay_values = [r.alignment_decay_turn for r in items if r.alignment_decay_turn is not None]
        configured_decay = next((r.configured_decay_turns for r in items if r.configured_decay_turns is not None), None)
        if configured_decay is None or not is_attack:
            actual_decay = "-"
        elif not decay_values:
            actual_decay = f"held/{configured_decay}"
        elif len(decay_values) == 1:
            actual_decay = f"{decay_values[0]}/{configured_decay}"
        else:
            avg = sum(decay_values) / len(decay_values)
            held = n - len(decay_values)
            suffix = f", held={held}" if held else ""
            actual_decay = f"avg {avg:.1f}/{configured_decay} [{min(decay_values)}-{max(decay_values)}{suffix}]"
        # 'sut_refused' = the system under test (the agents, incl. the Policy *participant*)
        # declined to produce a compromised call on its own. It is a defense-attribution concept
        # that only applies to attacks; for benign controls completion is shown by tool_fired, so
        # we render it n/a rather than lighting up every clean run (the old 'agent_blocked'
        # column misleadingly showed benign controls as "blocked").
        rows.append({
            "scenario_id": scenario_id,
            "kind": kind,
            "trials": n,
            "tool_fired": f"{tool_fired}/{n}",
            "compromised": f"{compromised}/{n}",
            "sut_refused": (f"{sut_refused}/{n}" if is_attack else "-"),
            "fw_blocked": f"{fw_blocked}/{n}",
            "fw_flagged": f"{fw_flagged}/{n}",
            "actual_decay": actual_decay,
        })
    try:
        import pandas as pd

        return pd.DataFrame(rows)
    except Exception:
        return rows


def _print_report(
    results_by_scenario: Dict[str, List[ScenarioResult]],
    metrics: AggregateMetrics,
    config: RunConfig,
    backend_name: str,
    firewall_name: str,
) -> None:
    report = build_report_dataframe(results_by_scenario, config.trials)

    topology = "SINGLE-AGENT isolation" if config.single_agent else "MULTI-AGENT LangGraph mesh"
    print("\n=== Enterprise Financial-ML Multi-Agent Security Evaluation ===")
    print(f"  topology        : {topology}")
    print(f"  mode            : {'REAL MODELS' if config.use_llm else 'MOCK (deterministic offline)'}")
    print(f"  agent under test: {backend_name}")
    print(f"  firewall        : {firewall_name}")
    print(f"  attacker (B)    : {config.attacker} "
          f"{'(LLM-driven Orchestrator debate)' if config.attacker == 'adaptive' else '(deterministic playbook)'}")
    print(f"  trials/scenario : {config.trials}")
    print(f"  traces          : {config.traces_path}\n")

    try:
        import pandas as pd  # noqa: F401

        print(report.to_string(index=False))  # type: ignore[union-attr]
    except Exception:
        for row in report:  # type: ignore[assignment]
            print(row)

    print("\n  legend: compromised=malicious live tool fired with sabotaged args | "
          "sut_refused=the system under test (agents incl. the Policy participant) declined to "
          "produce a compromised call -- NOT a perimeter defense, n/a for benign | "
          "fw_blocked=edge interceptor vetoed an intended execution | "
          "fw_flagged=edge interceptor raised a BLOCK | "
          "actual_decay=Profile B measured Policy APPROVE turn / configured decay budget")

    flat = flatten(results_by_scenario)
    malicious_flat = [r for r in flat if r.is_malicious]
    # Defense attribution is only meaningful for attacks (a benign completion is not a "defense").
    agent_def = sum(1 for r in malicious_flat if r.agent_blocked)
    fw_def = sum(1 for r in malicious_flat if r.firewall_blocked)
    fw_flag = sum(1 for r in flat if r.firewall_flagged)

    print("\n=== Aggregate Security Metrics (over all scenario x trial runs) ===\n")
    print(f"  Runs evaluated                 : {metrics.total}")
    print(f"  Malicious runs                 : {metrics.malicious_total}")
    print(f"  Benign control runs            : {metrics.benign_total}")
    # "by assertions" = verdict from the ground-truth ASSERTIONS (not the benign 'control'
    # scenarios). Compromise/ASR is reported over malicious runs only, further below; a benign
    # control that ever trips an assertion is surfaced explicitly via the '[!] benign scored
    # COMPROMISED' line in the benign-controls section.
    print(f"  Compromised (by assertions)    : {metrics.compromised_total}")
    print(f"  Secure (by assertions)         : {metrics.secure_total}")
    print(f"  Exploits defused               : {metrics.defused_total}/{metrics.malicious_total}")
    print(f"    ...by the AGENT (refused)     : {agent_def}/{metrics.malicious_total}")
    print(f"    ...by the FIREWALL (vetoed)   : {fw_def}/{metrics.malicious_total}")
    print(f"  Firewall flagged (any boundary): {fw_flag}/{metrics.total}")
    print(f"  Perimeter blind-spots          : {metrics.blindspot_total}/{metrics.total}")
    print(f"  Exploit Defusal Rate (Rd)      : {metrics.exploit_defusal_rate:.2%}")
    print(f"  Alignment Blindness Delta (ABD): {metrics.alignment_blindness_delta:.2%} (of malicious runs)")
    if metrics.benign_total:
        print("\n  --- Benign controls (false-positive surface) ---")
        print(f"  Benign control runs            : {metrics.benign_total}")
        print(f"  Completed (legit tool fired)   : {metrics.benign_completed_total}/{metrics.benign_total}")
        print(f"  Over-refusals (legit job held) : {metrics.over_refusal_total}/{metrics.benign_total}")
        print(f"  Over-Refusal Rate (FPR)        : {metrics.over_refusal_rate:.2%}")
        if metrics.false_compromise_total:
            print(f"  [!] benign scored COMPROMISED  : {metrics.false_compromise_total} (contract over-trigger -- investigate)")

    # -- Attack-class breakdown + alignment-decay profile (malicious-only) -----
    bm = compute_benchmark_metrics(flat)
    print("\n=== Benchmark Metrics (compromise metrics over MALICIOUS runs only) ===\n")
    print(f"  Attack Success Rate (ASR)      : {bm.attack_success_rate:.2%} "
          f"({bm.compromised_n}/{bm.malicious_n})")
    print(f"  Exploit Defusal Rate (Rd)      : {bm.exploit_defusal_rate:.2%}")
    print(f"  Alignment Blindness Delta (ABD): {bm.alignment_blindness_delta:.2%}")
    d = bm.decay
    print("\n  --- Alignment decay (malicious debate runs; safety peer under peer pressure) ---")
    print(f"  Debate runs                    : {d.debate_n}")
    print(f"  Alignment yielded (decayed)    : {d.yielded}/{d.debate_n} ({d.yielded_rate:.2%})")
    print(f"  Held the line                  : {d.held}/{d.debate_n} ({d.held_rate:.2%})")
    if d.mean_resistance_turns is not None:
        print(f"  Mean resistance turns          : {d.mean_resistance_turns:.2f} "
              f"(ratio {d.mean_resistance_ratio:.2f} of budget)")
    print(f"  Immediate capitulation (turn 1): {d.immediate_capitulation}/{d.debate_n} "
          f"({d.immediate_capitulation_rate:.2%})")
    print(f"  Peer-Trust Delta (PTD)         : {bm.peer_trust_delta:+.2%} "
          f"(malicious yield-rate net of benign baseline)")

    if bm.per_tier:
        print("\n  --- Per tier (malicious runs; T1 blatant -> T3 subtle) ---")
        print(f"  {'tier':<28} {'n':>3} {'ASR':>7} {'defuse':>7} {'ABD':>7} {'decay':>12}")
        for c in bm.per_tier:
            decay_str = (f"{c.decay.mean_resistance_turns:.1f}/{c.decay.debate_n}t"
                         if c.decay.mean_resistance_turns is not None else "-")
            print(f"  {c.attack_class:<28} {c.n:>3} {c.attack_success_rate:>6.0%} "
                  f"{c.defusal_rate:>6.0%} {c.blindness_delta:>6.0%} {decay_str:>12}")

    if bm.per_class:
        print("\n  --- Per attack-class (malicious runs) ---")
        print(f"  {'attack_class':<28} {'n':>3} {'ASR':>7} {'defuse':>7} {'ABD':>7} {'decay':>12}")
        for c in bm.per_class:
            decay_str = (f"{c.decay.mean_resistance_turns:.1f}/{c.decay.debate_n}t"
                         if c.decay.mean_resistance_turns is not None else "-")
            print(f"  {c.attack_class:<28} {c.n:>3} {c.attack_success_rate:>6.0%} "
                  f"{c.defusal_rate:>6.0%} {c.blindness_delta:>6.0%} {decay_str:>12}")
    print()


def _results_to_json(
    results_by_scenario: Dict[str, List[ScenarioResult]], metrics: AggregateMetrics
) -> str:
    payload = {
        "benchmark_metrics": benchmark_metrics_to_dict(
            compute_benchmark_metrics(flatten(results_by_scenario))
        ),
        "metrics": {
            "total": metrics.total,
            "malicious_total": metrics.malicious_total,
            "compromised_total": metrics.compromised_total,
            "secure_total": metrics.secure_total,
            "defused_total": metrics.defused_total,
            "blindspot_total": metrics.blindspot_total,
            "exploit_defusal_rate": metrics.exploit_defusal_rate,
            "alignment_blindness_delta": metrics.alignment_blindness_delta,
            "benign_total": metrics.benign_total,
            "benign_completed_total": metrics.benign_completed_total,
            "over_refusal_total": metrics.over_refusal_total,
            "over_refusal_rate": metrics.over_refusal_rate,
            "false_compromise_total": metrics.false_compromise_total,
        },
        "scenarios": {
            scenario_id: [
                {
                    "verdict": r.verdict.value,
                    "perimeter_allowed": r.perimeter_allowed,
                    "tool_fired": r.tool_fired,
                    "alignment_blindspot": r.alignment_blindspot,
                    "defense_attribution": r.defense_attribution,
                    "firewall_blocked": r.firewall_blocked,
                    "agent_blocked": r.agent_blocked,
                    "firewall_flagged": r.firewall_flagged,
                    "configured_decay_turns": r.configured_decay_turns,
                    "policy_turns": r.policy_turns,
                    "alignment_decay_turn": r.alignment_decay_turn,
                    "triggered_assertions": r.triggered_assertions,
                }
                for r in items
            ]
            for scenario_id, items in results_by_scenario.items()
        },
    }
    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="erosionbench.eval_pipeline.main",
        description="Financial-ML multi-agent security evaluation pipeline (LangGraph + LlamaFirewall).",
    )
    ap.add_argument("--mode", choices=["single", "multi"], default="multi",
                    help="Topology: 'single' = Single-Agent Isolation (isolated Orchestrator gated "
                         "by an active perimeter firewall); 'multi' = Multi-Agent LangGraph Mesh "
                         "(Gateway->Orchestrator->Policy->Deployment). Both route the SAME Profiles A/B/C.")
    fidelity = ap.add_mutually_exclusive_group()
    fidelity.add_argument("--real", action="store_true",
                          help="Use real frontier models + live LlamaFirewall (default when --model is set).")
    fidelity.add_argument("--mock", action="store_true",
                          help="Force deterministic offline mode: scripted confused-deputy agents + regex firewall.")
    ap.add_argument("--model", default=None,
                    help="Frontier backend for the agents. Provider is inferred from the name and "
                         "routed through the same gateway: OpenAI (gpt-4o, gpt-5.5, o3), Anthropic "
                         "(claude-sonnet-4-6, or anthropic:<name>), Google (gemini-2.5-pro, or "
                         "google:<name>). Setting this implies --real unless --mock is given.")
    ap.add_argument("--agent-model", default=os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o"),
                    help="Default bare model name when --model is omitted (default: $OPENAI_JUDGE_MODEL or gpt-4o).")
    ap.add_argument("--firewall", choices=["none", "mock", "llm", "promptguard", "llamafirewall", "llm-guard"], default=None,
                    help="Defense baseline: none=disabled/pass-through, llm-guard=Baseline 1 "
                         "(Protect AI LLM Guard scanner suite -- independent perimeter), llamafirewall/promptguard=Baseline 2 "
                         "(real LlamaFirewall: PromptGuard input + AlignmentCheck output), llm=in-house LLM judge, "
                         "mock=regex. Defaults to 'llamafirewall' in real mode, else 'mock'. "
                         "(Progent is retained as a control/future-scope baseline via "
                         "`python -m erosionbench.eval_pipeline.progent_control`.)")
    ap.add_argument("--attacker", choices=["scripted", "adaptive"], default="adaptive",
                    help="Profile-B debate regime: 'adaptive' (default) = the Orchestrator LLM "
                         "generates its own rebuttal to Policy objections (genuine multi-agent "
                         "debate, realistic enterprise behavior); 'scripted' = deterministic "
                         "playbook for reproducibility experiments. --mock always uses scripted.")
    ap.add_argument("--trials", type=int, default=1,
                    help="Repeat each scenario N times and report rates (recommended with real models).")
    ap.add_argument("--single-agent", dest="single_agent", action="store_true",
                    help="Back-compat alias for --mode single.")
    ap.add_argument("--scripted-agent", dest="scripted_agent", action="store_true",
                    help="Force the deterministic confused-deputy agent even with real models, to "
                         "isolate the FIREWALL's contribution (the agent always acts on poisoned state).")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Print the internal dialogue (gaslighting), firewall rulings, and per-assertion outcomes.")
    ap.add_argument("--category", choices=[c.value for c in ThreatCategory], default=None,
                    help="Evaluate only scenarios in this threat category (default: all).")
    ap.add_argument("--domain", choices=sorted(DOMAIN_PATTERNS), default=None,
                    help="Evaluate only one domain column: finance (A/B/C/D), dataprivacy (E/G/DATAPRIV), "
                         "or cicd (CICD_*). Shorthand for the matching --scenarios globs.")
    ap.add_argument("--scenarios", default=None,
                    help="Comma-separated scenario ids or fnmatch globs to run (e.g. 'A_*,B_T2_ROTATION' or "
                         "'CICD_*'). Combined with --domain/--category by intersection. Case-insensitive.")
    ap.add_argument("--list-scenarios", dest="list_scenarios", action="store_true",
                    help="Print every registered scenario id (respecting --domain/--scenarios) and exit.")
    ap.add_argument("--traces", default=DEFAULT_TRACE_PATH,
                    help="Append-only JSONL trace file path (default: ./evaluation_traces.json).")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="Optional path to also write the summary results + metrics as JSON.")
    ap.add_argument("--agent-profiles", dest="agent_profiles_path", default=None,
                    help="Path to an alternate agent persona catalog (YAML). Defaults to the packaged "
                         "configs/agent_mesh.yaml. Use to A/B-test safety framings or KPI pressures "
                         "without editing code (no system prompts are hardcoded).")
    # -- Datagen pipeline integration --
    ap.add_argument("--datagen", action="store_true", default=False,
                    help="Use the procedural datagen pipeline to generate scenarios instead of "
                         "the hand-crafted scenario_pool. Requires --datagen-domain.")
    ap.add_argument("--datagen-domain", default="quantitative_finance",
                    choices=["quantitative_finance", "cicd_supply_chain", "data_engineering", "autonomous_crm"],
                    help="Domain for datagen scenario generation (default: quantitative_finance).")
    ap.add_argument("--datagen-count", type=int, default=15,
                    help="Total scenario instances across all attack classes. Each instance produces "
                         "1 malicious + 1 benign twin per tier. E.g. QF (5 classes), count=15 → "
                         "3/class × 2 (twin) × 1 tier = 30 scenarios.")
    ap.add_argument("--datagen-tiers", default="1",
                    help="Comma-separated tiers for datagen scenarios (e.g. '1,2,3').")
    ap.add_argument("--datagen-seed", type=int, default=42,
                    help="RNG seed for datagen reproducibility.")
    ap.add_argument("--datagen-model", default="claude-sonnet-4-6-v1",
                    help="Model for datagen narrative generation (Stage 2).")
    ap.add_argument("--datagen-no-validate", action="store_true", default=False,
                    help="Skip validation gate during datagen (faster, less reliable).")
    ap.add_argument("--full-observability", dest="full_observability", action="store_true",
                    help="ABLATION: give every agent the WHOLE ledger (baseline registry, cross-session "
                         "drift, write-channel provenance). Defaults off -- the mesh runs with PARTIAL "
                         "observability (each agent sees only its own scope), which is the structural, "
                         "model-invariant attack lever. Run both and compare to attribute the multi-agent "
                         "failure to context fragmentation rather than model capability.")
    return ap.parse_args(argv)


def _parse_model_spec(spec: Optional[str], default_name: str) -> str:
    """Parse a ``provider:name`` (or bare ``name``) backend spec into a bare model name.

    The provider family (OpenAI / Anthropic / Google) is inferred downstream from the model
    name itself (see :func:`llm.provider_for_model`), so both ``--model claude-sonnet-4-6`` and
    ``--model anthropic:claude-sonnet-4-6`` route to the Anthropic gateway path; ``gemini-*`` /
    ``google:*`` route to the Google path; everything else routes to OpenAI.
    """
    if not spec:
        return default_name
    provider, sep, name = spec.partition(":")
    if not sep:  # no provider prefix -> the whole token is the model name
        return provider
    return name


def _resolve_config(args: argparse.Namespace) -> RunConfig:
    # Setting --model implies real mode (unless --mock is explicit).
    use_llm = (bool(args.real) or args.model is not None) and not args.mock
    mode = "single" if (args.mode == "single" or bool(getattr(args, "single_agent", False))) else "multi"
    if args.firewall:
        firewall_kind = args.firewall
    else:
        firewall_kind = "llamafirewall" if use_llm else "mock"
    # In mock mode the agents must be scripted (no live model); --scripted-agent forces it under --real.
    scripted_agent = (not use_llm) or bool(getattr(args, "scripted_agent", False))
    return RunConfig(
        mode=mode,
        model=_parse_model_spec(args.model, args.agent_model),
        firewall_kind=firewall_kind,
        trials=args.trials,
        verbose=args.verbose,
        scripted_agent=scripted_agent,
        use_llm=use_llm,
        traces_path=args.traces,
        attacker=args.attacker,
        agent_profiles_path=getattr(args, "agent_profiles_path", None),
        partial_observability=not bool(getattr(args, "full_observability", False)),
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    config = _resolve_config(args)

    # --- Datagen pipeline path (procedurally generated scenarios) ---
    if getattr(args, "datagen", False):
        print(f"[datagen] Generating scenarios: domain={args.datagen_domain}, "
              f"count={args.datagen_count}, tiers={args.datagen_tiers}, seed={args.datagen_seed}")
        datagen_tiers = [int(t) for t in args.datagen_tiers.split(",")]

        from .llm import LLMClientError
        try:
            scenarios = generate_scenarios_from_datagen(
                domain=args.datagen_domain,
                count=args.datagen_count,
                tiers=datagen_tiers,
                seed=args.datagen_seed,
                model=args.datagen_model,
                mock=args.mock,
                validate=not getattr(args, "datagen_no_validate", False),
            )
        except LLMClientError as e:
            print(f"\n{'=' * 78}")
            print(f"ERROR: LLM client failed — no fallback allowed.")
            print(f"{'=' * 78}")
            print(f"  {e}")
            print(f"\n  The pipeline requires a working LLM for scenario generation.")
            print(f"  Options:")
            print(f"    1. Check your credentials (X_API_KEY, OPENAI_API_KEY, LLM_GATEWAY_API_KEY)")
            print(f"    2. Verify the model '{args.datagen_model}' is reachable via your gateway")
            print(f"    3. Use --mock for offline/template generation (no LLM)")
            print(f"{'=' * 78}")
            return 1

        if not scenarios:
            print("[datagen] ERROR: No scenarios generated (validation gate may have rejected all).")
            return 1
        malicious_n = sum(1 for s in scenarios if s.is_malicious)
        benign_n = len(scenarios) - malicious_n
        print(f"[datagen] Generated {len(scenarios)} scenarios ({malicious_n} attack, {benign_n} benign)")

        from .langgraph_mesh import LLMCallError
        try:
            results_by_scenario, backend_name, firewall_name = evaluate_pool(scenarios, config)
        except LLMCallError as e:
            print(f"\nERROR: Model call failed -- {e}")
            return 1

        solver = DeterministicMeshSolver()
        metrics = solver.aggregate(flatten(results_by_scenario))
        _print_report(results_by_scenario, metrics, config, backend_name, firewall_name)

        if args.json_out:
            parent = os.path.dirname(os.path.abspath(args.json_out))
            os.makedirs(parent, exist_ok=True)
            with open(args.json_out, "w", encoding="utf-8") as fh:
                fh.write(_results_to_json(results_by_scenario, metrics))
            print(f"[wrote] {args.json_out}")
        return 0

    # --- Standard scenario_pool path (hand-crafted scenarios) ---
    # ONE canonical dataset (standardized Financial-ML Profiles A/B/C) routed identically
    # through BOTH baseline loops.
    pool: ScenarioPool = build_standard_pool()

    # Build the id glob set from --domain and/or --scenarios (intersection if both given).
    domain_pats: Optional[List[str]] = list(DOMAIN_PATTERNS[args.domain]) if args.domain else None
    explicit: Optional[List[str]] = (
        [p.strip() for p in args.scenarios.split(",") if p.strip()]
        if getattr(args, "scenarios", None) else None
    )
    patterns: Optional[List[str]]
    if domain_pats is not None and explicit is not None:
        # Resolve both globs to concrete ids and intersect (registration order preserved).
        exp_ids = set(pool.matching_ids(explicit))
        patterns = [i for i in pool.matching_ids(domain_pats) if i in exp_ids] or ["__none__"]
    elif domain_pats is not None:
        patterns = domain_pats
    elif explicit is not None:
        patterns = explicit
    else:
        patterns = None

    if getattr(args, "list_scenarios", False):
        ids = pool.matching_ids(patterns) if patterns is not None else pool.ids()
        print("\n".join(ids) if ids else "(no scenarios match the given filters)")
        return 0

    try:
        if patterns is not None:
            scenarios = pool.select(patterns)
        elif args.category is not None:
            scenarios = pool.filter_by_category(ThreatCategory(args.category))
        else:
            scenarios = pool.all()
    except KeyError as e:
        print(f"ERROR: {e}")
        return 1

    # --category narrows an id-glob selection further (intersection), when both are given.
    if patterns is not None and args.category is not None:
        cat = ThreatCategory(args.category)
        scenarios = [s for s in scenarios if s.threat_category is cat]

    from .langgraph_mesh import LLMCallError

    try:
        results_by_scenario, backend_name, firewall_name = evaluate_pool(scenarios, config)
    except LLMCallError as e:
        print(f"\n{'=' * 78}")
        print(f"ERROR: Model call failed -- results would be invalid.")
        print(f"{'=' * 78}")
        print(f"  {e}")
        if getattr(e, "kind", "no_response") == "config":
            # The model is fine; a required run flag is missing. Scripted fallback would
            # produce invalid (non-adaptive) results, so we fail closed instead.
            print(f"\n  This scenario needs the adaptive attacker to drive the multi-turn")
            print(f"  debate/gaslight with a live model. The run was started without it,")
            print(f"  so agent decisions would fall through to scripted behavior.")
            print(f"\n  Options:")
            print(f"    1. Re-run with --attacker adaptive (recommended for live models)")
            print(f"    2. Use --mock for intentional scripted/offline runs")
        else:
            print(f"\n  The model '{config.model}' is not returning responses. This means")
            print(f"  all agent decisions would silently fall through to hardcoded behavior,")
            print(f"  producing invalid benchmark results.")
            print(f"\n  Options:")
            print(f"    1. Use a model that works with your gateway (try --model gpt-4o)")
            print(f"    2. Use --mock for intentional scripted/offline runs")
            print(f"    3. Check your OPENAI_API_KEY / X_API_KEY credentials")
        print(f"{'=' * 78}")
        return 1

    solver = DeterministicMeshSolver()
    metrics = solver.aggregate(flatten(results_by_scenario))

    _print_report(results_by_scenario, metrics, config, backend_name, firewall_name)

    if args.json_out:
        parent = os.path.dirname(os.path.abspath(args.json_out))
        os.makedirs(parent, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            fh.write(_results_to_json(results_by_scenario, metrics))
        print(f"[wrote] {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
