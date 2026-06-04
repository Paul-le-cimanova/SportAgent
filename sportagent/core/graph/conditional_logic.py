"""Conditional routing for the SportAgent graph.

Three kinds of branch decisions:

1. **Analyst tool loop** — after a tool-bound analyst runs, route back to its
   tool node while it keeps calling tools, otherwise to its Msg-Clear node.
2. **Research debate** — alternate Bull ↔ Bear until
   ``count >= 2 * max_debate_rounds``, then hand to the Research Manager.
3. **Risk debate** — rotate Aggressive → Conservative → Neutral until
   ``count >= 3 * max_risk_rounds``, then hand to the Decision Manager.

Node-name constants are shared with ``setup.py`` so the labels stay in sync.
"""

from __future__ import annotations

# --- Shared node-name constants ----------------------------------------------

BULL_RESEARCHER = "Bull Researcher"
BEAR_RESEARCHER = "Bear Researcher"
RESEARCH_MANAGER = "Research Manager"
TRADER = "Trader"
AGGRESSIVE = "Aggressive Debator"
CONSERVATIVE = "Conservative Debator"
NEUTRAL = "Neutral Debator"
DECISION_MANAGER = "Decision Manager"


class ConditionalLogic:
    """Branch decisions for analyst loops + research/risk debates."""

    def __init__(self, max_debate_rounds: int = 1, max_risk_rounds: int = 1):
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_rounds = max_risk_rounds

    # --- Analyst tool loops --------------------------------------------------

    @staticmethod
    def _has_tool_calls(state) -> bool:
        messages = state.get("messages") or []
        if not messages:
            return False
        return bool(getattr(messages[-1], "tool_calls", None))

    def make_should_continue_analyst(self, tool_node: str, clear_node: str):
        """Return a router: tool node while tool-calling, else the Msg-Clear node."""

        def router(state) -> str:
            return tool_node if self._has_tool_calls(state) else clear_node

        return router

    # --- Research debate (Bull <-> Bear) -------------------------------------

    def should_continue_debate(self, state) -> str:
        """Alternate Bull/Bear until the round budget is spent, then judge."""
        debate = state.get("investment_debate_state", {}) or {}
        count = debate.get("count", 0)
        if count >= 2 * self.max_debate_rounds:
            return RESEARCH_MANAGER
        current = debate.get("current_response", "") or ""
        # If the Bull just spoke, the Bear answers next; otherwise Bull.
        if current.startswith("Bull"):
            return BEAR_RESEARCHER
        return BULL_RESEARCHER

    # --- Risk debate (Aggressive -> Conservative -> Neutral) -----------------

    def should_continue_risk(self, state) -> str:
        """Rotate the three risk voices until the budget is spent, then decide."""
        risk = state.get("risk_debate_state", {}) or {}
        count = risk.get("count", 0)
        if count >= 3 * self.max_risk_rounds:
            return DECISION_MANAGER
        speaker = risk.get("latest_speaker", "") or ""
        if speaker.startswith("Aggressive"):
            return CONSERVATIVE
        if speaker.startswith("Conservative"):
            return NEUTRAL
        return AGGRESSIVE