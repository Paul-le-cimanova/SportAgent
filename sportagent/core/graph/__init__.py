"""SportAgent graph layer — LangGraph pipeline assembly + orchestration.

Re-exports the public entry points so callers can do:

    from sportagent.core.graph import SportAgentGraph

The graph wires the analyst → research-debate → trader → risk-debate →
decision pipeline over a shared ``GameState`` (see design docs 03 & 06).
"""

from typing import TYPE_CHECKING, Any

from sportagent.core.graph.signal_processing import parse_action, process_signal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sportagent.core.graph.propagation import create_initial_state
    from sportagent.core.graph.sport_graph import SportAgentGraph

__all__ = [
    "SportAgentGraph",
    "create_initial_state",
    "parse_action",
    "process_signal",
]


def __getattr__(name: str) -> Any:
    """Lazily import the heavy graph entry points on first access.

    Keeps ``signal_processing`` (stdlib + pydantic only) importable without
    pulling in langchain/langgraph, which are only needed to actually build
    and run the graph.
    """
    if name == "SportAgentGraph":
        from sportagent.core.graph.sport_graph import SportAgentGraph

        return SportAgentGraph
    if name == "create_initial_state":
        from sportagent.core.graph.propagation import create_initial_state

        return create_initial_state
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
