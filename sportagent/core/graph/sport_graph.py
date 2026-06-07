"""SportAgentGraph — top-level orchestrator for a single market analysis.

``SportAgentGraph(config, debug).analyze(query)`` runs the full pipeline:

1. Resolve the sport adapter (default ``"nba"``; importing the adapter module
   self-registers it + its data tools).
2. ``adapter.resolve_market(query)`` → a ``MarketRef`` (Kalshi contract(s),
   teams, date, outcome structure).
3. Build the deterministic ``game_context`` (anti-hallucination identity).
4. Pre-compute the verified-odds snapshot (source of truth for prices).
5. Inject prior settled-outcome lessons as ``past_context``.
6. ``create_initial_state`` → run the compiled graph.
7. Append a pending decision-log entry and return
   ``(final_state, final_recommendation)``.

Before the run, pending settled entries are resolved (Brier reflection) so the
freshest lessons feed this run's Decision Manager.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Any, Dict, Optional, Tuple

from sportagent.core.agents.utils import memory
from sportagent.core.agents.utils.agent_utils import build_game_context
from sportagent.core.dataflows import odds_validator
from sportagent.core.dataflows.config import get_config, set_config
from sportagent.core.graph.conditional_logic import ConditionalLogic
from sportagent.core.graph.propagation import create_initial_state
from sportagent.core.graph.setup import GraphSetup
from sportagent.core.graph.signal_processing import parse_action
from sportagent.core.llm_clients.factory import create_deep_and_quick

logger = logging.getLogger(__name__)


def _get_adapter(sport: str):
    """Resolve a sport adapter, importing its module to self-register it."""
    from sportagent.sports.base import get_adapter

    adapter = get_adapter(sport)
    if adapter is not None:
        return adapter

    # Import the adapter module for the side-effect of registration.
    try:
        __import__(f"sportagent.sports.{sport}.adapter")
    except Exception as exc:  # noqa: BLE001 — fail open to a clear error
        logger.warning("Could not import adapter for sport %r: %s", sport, exc)
    return get_adapter(sport)


class SportAgentGraph:
    """Compiles + runs the SportAgent pipeline for one market query."""

    def __init__(self, config: Optional[dict] = None, debug: bool = False):
        self.config = config or get_config()
        # Make this run's config the active runtime config for downstream tools.
        set_config(self.config)
        self.debug = debug

        self.deep_llm, self.quick_llm = create_deep_and_quick(self.config)
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config.get("max_debate_rounds", 1),
            max_risk_rounds=self.config.get("max_risk_rounds", 1),
        )

    # --- Public API ----------------------------------------------------------

    def analyze(
        self, query: str, sport: str = "nba", game_date: Optional[str] = None
    ) -> Tuple[Dict[str, Any], str]:
        """Run the full pipeline for ``query`` and return (state, recommendation).

        Args:
            query: A Kalshi market ticker (preferred, exact) or a free-text
                matchup like ``"Spurs @ Knicks"``.
            sport: Sport key selecting the adapter (default ``"nba"``).
            game_date: Optional YYYY-MM-DD date of the game (from the wizard's
                schedule selection). Used to pick the right-dated Kalshi market
                and to stamp the report with the date the user actually picked.

        Returns:
            ``(final_state, final_recommendation_markdown)``.
        """
        adapter = _get_adapter(sport)
        if adapter is None:
            msg = f"No adapter registered for sport {sport!r}."
            logger.error(msg)
            return {}, f"<error: {msg}>"

        # Resolve before this run's lessons are gathered so settled prior calls
        # feed the Decision Manager.
        self._resolve_prior_outcomes()

        market_ref = self._resolve(adapter, query, game_date)
        game_context = build_game_context(
            market_ticker=getattr(market_ref, "market_ticker", ""),
            target_team=getattr(market_ref, "target_team", ""),
            home_team=getattr(market_ref, "home_team", ""),
            away_team=getattr(market_ref, "away_team", ""),
            game_date=getattr(market_ref, "game_date", ""),
        )

        verified_odds = self._build_verified_odds(market_ref, adapter)

        past_context = self._gather_past_context(market_ref)

        initial_state = create_initial_state(
            market_ref,
            game_context,
            past_context,
            verified_odds=verified_odds,
        )

        graph = self._compile_graph(adapter)
        final_state = self._run(graph, initial_state)

        final_recommendation = final_state.get("final_recommendation", "")
        self._log_decision(market_ref, final_state)

        return final_state, final_recommendation

    def analyze_stream(
        self, query: str, sport: str = "nba", game_date: Optional[str] = None
    ):
        """Run the pipeline as a generator, yielding node-level updates.

        Yields ``(node_name, state_delta)`` after each node so a live UI can
        flip the progress table pending->running->done. The final item yielded
        is ``("__final__", (final_state, recommendation))``.
        """
        adapter = _get_adapter(sport)
        if adapter is None:
            msg = f"No adapter registered for sport {sport!r}."
            logger.error(msg)
            yield "__final__", ({}, f"<error: {msg}>")
            return

        self._resolve_prior_outcomes()
        market_ref = self._resolve(adapter, query, game_date)
        game_context = build_game_context(
            market_ticker=getattr(market_ref, "market_ticker", ""),
            target_team=getattr(market_ref, "target_team", ""),
            home_team=getattr(market_ref, "home_team", ""),
            away_team=getattr(market_ref, "away_team", ""),
            game_date=getattr(market_ref, "game_date", ""),
        )
        verified_odds = self._build_verified_odds(market_ref, adapter)
        past_context = self._gather_past_context(market_ref)
        initial_state = create_initial_state(
            market_ref, game_context, past_context, verified_odds=verified_odds
        )

        graph = self._compile_graph(adapter)
        recur_limit = self.config.get("max_recur_limit", 100)
        accumulated: Dict[str, Any] = dict(initial_state)
        try:
            # Node-level streaming ("updates"): each chunk is {node_name: delta}.
            # The live UI shows progress + the latest completed section; motion
            # between agents comes from the spinner + steady refresh, so we do
            # NOT use token ("messages") streaming here (it changed message
            # content to block-lists and broke downstream string joins).
            for chunk in graph.stream(
                initial_state, {"recursion_limit": recur_limit}
            ):
                if not isinstance(chunk, dict):
                    continue
                for node_name, delta in chunk.items():
                    if isinstance(delta, dict):
                        accumulated.update(delta)
                    yield node_name, delta
        except Exception:  # surface full traceback to the log, fail soft to UI
            logger.exception("Streamed graph run failed")
            import traceback as _tb
            accumulated["error"] = _tb.format_exc()
            accumulated["final_recommendation"] = (
                "<error: graph run failed — see the run log "
                "(~/.sportagent/logs/) for the full traceback>"
            )

        final_recommendation = accumulated.get("final_recommendation", "")
        self._log_decision(market_ref, accumulated)
        yield "__final__", (accumulated, final_recommendation)

    # --- Internals -----------------------------------------------------------

    def _resolve(self, adapter, query: str, game_date: Optional[str]):  # noqa: D401
        """Resolve the market, passing the wizard-selected date when supported.

        Older adapters may not accept a ``game_date`` kwarg, so fall back
        gracefully to the two-arg signature.
        """
        try:
            return adapter.resolve_market(query, self.config, game_date=game_date)
        except TypeError:
            return adapter.resolve_market(query, self.config)

    def _compile_graph(self, adapter):
        key_factors = ""
        try:
            key_factors = adapter.key_factors_prompt()
        except Exception as exc:  # noqa: BLE001
            logger.warning("key_factors_prompt failed: %s", exc)
        setup = GraphSetup(
            deep_llm=self.deep_llm,
            quick_llm=self.quick_llm,
            conditional_logic=self.conditional_logic,
            key_factors_prompt=key_factors,
        )
        workflow = setup.setup_graph()
        return workflow.compile()

    def _run(self, graph, initial_state):
        recur_limit = self.config.get("max_recur_limit", 100)
        try:
            return graph.invoke(initial_state, {"recursion_limit": recur_limit})
        except Exception:  # surface full traceback to the log, fail soft
            logger.exception("Graph run failed")
            import traceback as _tb
            failed = dict(initial_state)
            failed["error"] = _tb.format_exc()
            failed["final_recommendation"] = (
                "<error: graph run failed — see the run log "
                "(~/.sportagent/logs/) for the full traceback>"
            )
            return failed

    def _build_verified_odds(self, market_ref, adapter) -> str:
        try:
            sportsbook_key = getattr(market_ref, "sportsbook_key", "") or adapter.sportsbook_key()
            return odds_validator.build_verified_odds_snapshot(
                market_ticker=getattr(market_ref, "market_ticker", ""),
                target_team=getattr(market_ref, "target_team", ""),
                home_team=getattr(market_ref, "home_team", ""),
                away_team=getattr(market_ref, "away_team", ""),
                sport_key=sportsbook_key,
                config=self.config,
            )
        except Exception as exc:  # noqa: BLE001 — fail open
            logger.warning("Verified-odds snapshot failed: %s", exc)
            return ""

    def _gather_past_context(self, market_ref) -> str:
        try:
            teams = [
                getattr(market_ref, "home_team", ""),
                getattr(market_ref, "away_team", ""),
                getattr(market_ref, "target_team", ""),
            ]
            return memory.get_past_context(
                market_ticker=getattr(market_ref, "market_ticker", ""),
                teams=[t for t in teams if t],
                config=self.config,
            )
        except Exception as exc:  # noqa: BLE001 — fail open
            logger.warning("get_past_context failed: %s", exc)
            return ""

    def _resolve_prior_outcomes(self) -> None:
        try:
            from sportagent.core.graph.reflection import resolve_pending_entries

            resolved = resolve_pending_entries(self.quick_llm, self.config)
            if resolved:
                logger.info("Resolved %d settled prior decision(s).", resolved)
        except Exception as exc:  # noqa: BLE001 — fail open
            logger.warning("Prior-outcome resolution failed: %s", exc)

    def _log_decision(self, market_ref, final_state: Dict[str, Any]) -> None:
        recommendation = final_state.get("final_recommendation", "")
        if not recommendation or recommendation.startswith("<error"):
            return
        try:
            action = parse_action(recommendation)
            est, impl = _extract_probs(recommendation)
            date = getattr(market_ref, "game_date", "") or _dt.date.today().isoformat()
            memory.append_pending_entry(
                date=date,
                market_ticker=getattr(market_ref, "market_ticker", ""),
                action=action,
                estimated_probability=est,
                implied_probability=impl,
                recommendation_markdown=recommendation,
                config=self.config,
            )
        except Exception as exc:  # noqa: BLE001 — fail open
            logger.warning("Decision-log append failed: %s", exc)


def _extract_probs(recommendation: str) -> Tuple[float, float]:
    """Parse estimated + implied probabilities from the rendered recommendation.

    Matches the winner-first betting-view labels ("Estimated target-YES
    probability" / "Implied probability"), with a looser fallback for older
    "Estimated Probability" / "Implied Probability" labels.
    """
    import re

    def _grab(*labels: str) -> float:
        for label in labels:
            m = re.search(
                label + r"[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*%", recommendation, re.IGNORECASE
            )
            if m:
                return min(1.0, max(0.0, float(m.group(1)) / 100.0))
        return 0.5

    est = _grab("Estimated target-YES probability", "Estimated Probability")
    impl = _grab("Implied probability", "Implied Probability")
    return est, impl
