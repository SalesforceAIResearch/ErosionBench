"""Side utility: run the **Progent** privilege-control baseline (control / future scope).

Progent is intentionally NOT one of the two primary perimeter baselines (those are LlamaFirewall
and Protect AI's LLM Guard, selected via ``main.py --firewall``). It is retained here as a
*control* baseline and a hook for future work, because a deterministic least-privilege monitor
answers a different, complementary question than a perimeter text scanner.

Its most useful diagnostic in this harness is the **false-positive surface**: does a strict,
default-deny least-privilege policy over-block *legitimate* operations? This script therefore
runs Progent over the benign **control** scenarios by default and reports the over-refusal rate;
pass ``--include-attacks`` to also see its exploit-defusal view for reference.

Examples
--------
    # Offline, controls only (FPR surface):
    python -m erosionbench.eval_pipeline.progent_control --mock

    # Live, controls + attacks (full reference view):
    python -m erosionbench.eval_pipeline.progent_control --model gpt-4o --include-attacks
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from .main import RunConfig, evaluate_pool, flatten
from .scenario_pool import build_standard_pool
from .schemas import EnterpriseScenario
from .solver import DeterministicMeshSolver


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="erosionbench.eval_pipeline.progent_control",
        description="Run the Progent privilege-control baseline over control scenarios (FPR surface).",
    )
    ap.add_argument("--mode", choices=["single", "multi"], default="multi")
    ap.add_argument("--model", default="gpt-4o", help="Agent backend when not --mock.")
    ap.add_argument("--mock", action="store_true",
                    help="Offline: scripted agents + scripted (deterministic) Progent policy.")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--include-attacks", dest="include_attacks", action="store_true",
                    help="Also run malicious scenarios to show Progent's defusal view (reference).")
    ap.add_argument("--traces", default="progent_control_traces.json",
                    help="Append-only JSONL trace path (kept separate from the primary baselines').")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    use_llm = not args.mock
    config = RunConfig(
        mode=args.mode,
        model=args.model,
        firewall_kind="progent",
        trials=args.trials,
        verbose=False,
        scripted_agent=not use_llm,
        use_llm=use_llm,
        traces_path=args.traces,
        attacker="scripted" if args.mock else "adaptive",
    )

    pool = build_standard_pool()
    scenarios: List[EnterpriseScenario] = pool.all()
    if not args.include_attacks:
        scenarios = [s for s in scenarios if not s.is_malicious]

    results_by_scenario, backend_name, firewall_name = evaluate_pool(scenarios, config)
    metrics = DeterministicMeshSolver().aggregate(flatten(results_by_scenario))

    print("\n=== Progent (control/future-scope baseline) ===")
    print(f"  topology      : {'single' if config.single_agent else 'multi'}")
    print(f"  firewall      : {firewall_name}")
    print(f"  agent backend : {backend_name}")
    print(f"  trials/scen.  : {config.trials}\n")
    if metrics.benign_total:
        print("  --- Benign controls (false-positive surface) ---")
        print(f"  control runs              : {metrics.benign_total}")
        print(f"  completed (legit fired)   : {metrics.benign_completed_total}/{metrics.benign_total}")
        print(f"  over-refusals (held)      : {metrics.over_refusal_total}/{metrics.benign_total}")
        print(f"  Over-Refusal Rate (FPR)   : {metrics.over_refusal_rate:.2%}")
    if args.include_attacks and metrics.malicious_total:
        print("\n  --- Malicious (defusal reference) ---")
        print(f"  exploits defused (Rd)     : {metrics.defused_total}/{metrics.malicious_total} "
              f"({metrics.exploit_defusal_rate:.2%})")
        print(f"  perimeter blind-spots     : {metrics.blindspot_total}/{metrics.total}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
