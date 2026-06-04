"""StateGraph assembly for the SportAgent pipeline.

Wires the full graph over ``GameState``:

    START → Odds Analyst → (tools loop) → Msg-Clear
          → Stats Analyst → (tools loop) → Msg-Clear
          → News/Injury Analyst → (tools loop) → Msg-Clear
          → Sentiment Analyst (structured, no tools) → Msg-Clear
          → Bull ↔ Bear debate → Research Manager
          → Trader
          → Aggressive → Conservative → Neutral risk debate → Decision Manager
          → END

The three tool-bound analysts (odds/stats/news) each get a ``ToolNode`` and a
self-loop driven by ``ConditionalLogic``; the Sentiment Analyst pre-fetches its
data internally and emits structured output, so it has no tool node. The deep
LLM tier serves only the Research Manager + Decision Manager; the quick tier
serves everyone else (see design docs 03 & 06).
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from sportagent.core.agents import (
    create_aggressive_debator,
    create_bear_researcher,
    create_bull_researcher,
    create_conservative_debator,
    create_decision_manager,
    create_neutral_debator,
    create_news_injury_analyst,
    create_odds_analyst,
    create_research_manager,
    create_sentiment_analyst,
    create_stats_analyst,
    create_trader,
)
from sportagent.core.agents.utils.agent_utils import create_msg_delete
from sportagent.core.agents.utils.game_state import GameState
from sportagent.core.agents.utils.tools import NEWS_TOOLS, ODDS_TOOLS, STATS_TOOLS
from sportagent.core.graph.conditional_logic import (
    AGGRESSIVE,
    BEAR_RESEARCHER,
    BULL_RESEARCHER,
    CONSERVATIVE,
    DECISION_MANAGER,
    NEUTRAL,
    RESEARCH_MANAGER,
    TRADER,
    ConditionalLogic,
)

# Analyst node-name constants.
ODDS_ANALYST = "Odds Analyst"
STATS_ANALYST = "Stats Analyst"
NEWS_ANALYST = "News/Injury Analyst"
SENTIMENT_ANALYST = "Sentiment Analyst"


class GraphSetup:
    """Builds + compiles the SportAgent ``StateGraph``."""

    def __init__(
        self,
        deep_llm: Any,
        quick_llm: Any,
        conditional_logic: ConditionalLogic,
        key_factors_prompt: str = "",
    ):
        self.deep_llm = deep_llm
        self.quick_llm = quick_llm
        self.conditional_logic = conditional_logic
        self.key_factors_prompt = key_factors_prompt

    def setup_graph(self):
        """Construct the workflow graph (call ``.compile()`` on the result)."""
        cl = self.conditional_logic

        # --- Agent nodes -----------------------------------------------------
        odds_node = create_odds_analyst(self.quick_llm)
        stats_node = create_stats_analyst(self.quick_llm, self.key_factors_prompt)
        news_node = create_news_injury_analyst(self.quick_llm)
        sentiment_node = create_sentiment_analyst(self.quick_llm)

        bull_node = create_bull_researcher(self.quick_llm)
        bear_node = create_bear_researcher(self.quick_llm)
        research_manager_node = create_research_manager(self.deep_llm)
        trader_node = create_trader(self.quick_llm)
        aggressive_node = create_aggressive_debator(self.quick_llm)
        conservative_node = create_conservative_debator(self.quick_llm)
        neutral_node = create_neutral_debator(self.quick_llm)
        decision_manager_node = create_decision_manager(self.deep_llm)

        workflow = StateGraph(GameState)

        # --- Tool-bound analysts: agent + tool node + Msg-Clear --------------
        # (analyst_name, agent_node, tools, tool_node_name, clear_node_name)
        tool_analysts = [
            (ODDS_ANALYST, odds_node, ODDS_TOOLS, "tools_odds", "Msg Clear Odds"),
            (STATS_ANALYST, stats_node, STATS_TOOLS, "tools_stats", "Msg Clear Stats"),
            (NEWS_ANALYST, news_node, NEWS_TOOLS, "tools_news", "Msg Clear News"),
        ]
        for name, node, tools, tool_node, clear_node in tool_analysts:
            workflow.add_node(name, node)
            workflow.add_node(tool_node, ToolNode(tools))
            workflow.add_node(clear_node, create_msg_delete())

        # Sentiment analyst (structured, no tools) + its Msg-Clear.
        workflow.add_node(SENTIMENT_ANALYST, sentiment_node)
        sentiment_clear = "Msg Clear Sentiment"
        workflow.add_node(sentiment_clear, create_msg_delete())

        # --- Research / trader / risk / decision nodes -----------------------
        workflow.add_node(BULL_RESEARCHER, bull_node)
        workflow.add_node(BEAR_RESEARCHER, bear_node)
        workflow.add_node(RESEARCH_MANAGER, research_manager_node)
        workflow.add_node(TRADER, trader_node)
        workflow.add_node(AGGRESSIVE, aggressive_node)
        workflow.add_node(CONSERVATIVE, conservative_node)
        workflow.add_node(NEUTRAL, neutral_node)
        workflow.add_node(DECISION_MANAGER, decision_manager_node)

        # --- Analyst chain edges ---------------------------------------------
        workflow.add_edge(START, ODDS_ANALYST)

        # Tool-bound analysts: self-loop on tool calls, else clear → next.
        clear_targets = {
            "Msg Clear Odds": STATS_ANALYST,
            "Msg Clear Stats": NEWS_ANALYST,
            "Msg Clear News": SENTIMENT_ANALYST,
        }
        for name, _node, _tools, tool_node, clear_node in tool_analysts:
            workflow.add_conditional_edges(
                name,
                cl.make_should_continue_analyst(tool_node, clear_node),
                [tool_node, clear_node],
            )
            workflow.add_edge(tool_node, name)
            workflow.add_edge(clear_node, clear_targets[clear_node])

        # Sentiment analyst → its clear → Bull (start of research debate).
        workflow.add_edge(SENTIMENT_ANALYST, sentiment_clear)
        workflow.add_edge(sentiment_clear, BULL_RESEARCHER)

        # --- Research debate (Bull <-> Bear → Research Manager) --------------
        workflow.add_conditional_edges(
            BULL_RESEARCHER,
            cl.should_continue_debate,
            [BEAR_RESEARCHER, RESEARCH_MANAGER],
        )
        workflow.add_conditional_edges(
            BEAR_RESEARCHER,
            cl.should_continue_debate,
            [BULL_RESEARCHER, RESEARCH_MANAGER],
        )

        # --- Research Manager → Trader → risk debate -------------------------
        workflow.add_edge(RESEARCH_MANAGER, TRADER)
        workflow.add_edge(TRADER, AGGRESSIVE)

        # Risk debate rotation → Decision Manager.
        workflow.add_conditional_edges(
            AGGRESSIVE,
            cl.should_continue_risk,
            [CONSERVATIVE, DECISION_MANAGER],
        )
        workflow.add_conditional_edges(
            CONSERVATIVE,
            cl.should_continue_risk,
            [NEUTRAL, DECISION_MANAGER],
        )
        workflow.add_conditional_edges(
            NEUTRAL,
            cl.should_continue_risk,
            [AGGRESSIVE, DECISION_MANAGER],
        )

        # --- Terminal --------------------------------------------------------
        workflow.add_edge(DECISION_MANAGER, END)

        return workflow