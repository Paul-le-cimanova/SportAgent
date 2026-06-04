"""SportAgent agents: analysts, researchers, trader, risk, managers.

Re-exports every agent factory so the graph can import them from one place:

    from sportagent.core.agents import create_odds_analyst, create_trader, ...
"""

from sportagent.core.agents.analysts.news_injury_analyst import (
    create_news_injury_analyst,
)
from sportagent.core.agents.analysts.odds_analyst import create_odds_analyst
from sportagent.core.agents.analysts.sentiment_analyst import create_sentiment_analyst
from sportagent.core.agents.analysts.stats_analyst import create_stats_analyst
from sportagent.core.agents.managers.decision_manager import create_decision_manager
from sportagent.core.agents.managers.research_manager import create_research_manager
from sportagent.core.agents.researchers.bear_researcher import create_bear_researcher
from sportagent.core.agents.researchers.bull_researcher import create_bull_researcher
from sportagent.core.agents.risk.aggressive_debator import create_aggressive_debator
from sportagent.core.agents.risk.conservative_debator import (
    create_conservative_debator,
)
from sportagent.core.agents.risk.neutral_debator import create_neutral_debator
from sportagent.core.agents.trader.trader import create_trader

__all__ = [
    "create_odds_analyst",
    "create_stats_analyst",
    "create_news_injury_analyst",
    "create_sentiment_analyst",
    "create_bull_researcher",
    "create_bear_researcher",
    "create_research_manager",
    "create_decision_manager",
    "create_trader",
    "create_aggressive_debator",
    "create_conservative_debator",
    "create_neutral_debator",
]
