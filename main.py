"""SportAgent quick-start entry point.

Run a single market analysis without the full CLI:

    python main.py "Spurs @ Knicks"
    python main.py KXNBAGAME-26JUN02SASNYK-SAS

Loads ``.env`` (via the package import), runs an environment preflight, and —
if all required keys are present — builds ``SportAgentGraph`` and prints the
final recommendation plus key state (verified odds, edge thesis, position).

For the richer interface (``setup`` / ``doctor`` / ``analyze``) use the console
script: ``sportagent <command>``.
"""

from __future__ import annotations

import sys

# Importing the package loads .env so config + data clients see the keys.
import sportagent  # noqa: F401
from sportagent import onboarding


def _print_report(results) -> None:
    """Plain-text preflight report (no rich dependency required)."""
    for r in results:
        flag = "*" if r.required else " "
        hint = f"  — {r.hint}" if r.hint else ""
        print(f"  [{flag}] {r.status:<8} {r.name}{hint}")


def main(argv: list[str] | None = None) -> int:
    """Quick-start: preflight, then analyze ``argv[1]`` (or prompt)."""
    argv = list(sys.argv if argv is None else argv)
    query = argv[1] if len(argv) > 1 else ""
    sport = argv[2] if len(argv) > 2 else "nba"

    results = onboarding.check_environment(live=False)
    missing = onboarding.required_missing(results)
    if missing:
        print("SportAgent is not configured yet:\n")
        _print_report(results)
        print(
            "\nRun `sportagent setup` (or edit .env) to add the missing keys, "
            "then `sportagent doctor --live` to verify."
        )
        return 1

    if not query:
        print('Usage: python main.py "<ticker or matchup>" [sport]')
        print('Example: python main.py "Spurs @ Knicks" nba')
        return 2

    # Imported lazily so preflight failures don't pay the heavy import cost.
    from sportagent.core.graph.sport_graph import SportAgentGraph

    print(f"\nAnalyzing: {query}  (sport={sport})\n")
    graph = SportAgentGraph()
    state, recommendation = graph.analyze(query, sport=sport)

    verified = state.get("verified_odds", "")
    thesis = state.get("investment_plan", "")
    position = state.get("trader_position_plan", "")

    if verified:
        print("=== Verified odds ===")
        print(verified, "\n")
    if thesis:
        print("=== Edge thesis ===")
        print(thesis, "\n")
    if position:
        print("=== Position proposal ===")
        print(position, "\n")

    print("=== Final recommendation ===")
    print(recommendation or "<no recommendation produced>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())