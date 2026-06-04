"""rich.Live streaming UI for a SportAgent run (doc 10 §3 port).

Renders a live layout while ``SportAgentGraph.analyze_stream`` runs:

  - Header     : welcome banner
  - Progress   : Team/Agent/Status table (pending -> in_progress -> completed)
  - Analysis   : the latest report section as markdown
  - Footer     : Agents / Reports / elapsed stats

Teams (doc 10 §6 mapping):
  Analyst Team  [Odds / Stats / News-Injury / Sentiment]
  Research Team [Bull / Bear / Research Manager]
  Trader        [Trader]
  Risk Mgmt     [Aggressive / Neutral / Conservative]
  Decision      [Decision Manager]

Degrades gracefully: if rich.Live can't run, the caller falls back to the
non-streaming path.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.layout import Layout
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich import box

# Graph node name -> (team, display label). Node names mirror setup.py.
_NODE_TO_AGENT: Dict[str, Tuple[str, str]] = {
    "Odds Analyst": ("Analyst Team", "Odds Analyst"),
    "Stats Analyst": ("Analyst Team", "Stats Analyst"),
    "News/Injury Analyst": ("Analyst Team", "News-Injury Analyst"),
    "Sentiment Analyst": ("Analyst Team", "Sentiment Analyst"),
    "Bull Researcher": ("Research Team", "Bull Researcher"),
    "Bear Researcher": ("Research Team", "Bear Researcher"),
    "Research Manager": ("Research Team", "Research Manager"),
    "Trader": ("Trader", "Trader"),
    "Aggressive Debator": ("Risk Management", "Aggressive"),
    "Conservative Debator": ("Risk Management", "Conservative"),
    "Neutral Debator": ("Risk Management", "Neutral"),
    "Decision Manager": ("Decision", "Decision Manager"),
}

# Ordered teams + their agents for the progress panel.
_TEAMS: List[Tuple[str, List[str]]] = [
    ("Analyst Team", ["Odds Analyst", "Stats Analyst", "News-Injury Analyst", "Sentiment Analyst"]),
    ("Research Team", ["Bull Researcher", "Bear Researcher", "Research Manager"]),
    ("Trader", ["Trader"]),
    ("Risk Management", ["Aggressive", "Neutral", "Conservative"]),
    ("Decision", ["Decision Manager"]),
]

# State key a node writes -> report section title (for the Analysis panel).
_REPORT_KEYS: List[Tuple[str, str]] = [
    ("odds_report", "Odds Analysis"),
    ("stats_report", "Stats Analysis"),
    ("news_report", "News & Injury Analysis"),
    ("sentiment_report", "Sentiment Analysis"),
    ("investment_plan", "Research Team Decision"),
    ("trader_position_plan", "Trader Position"),
    ("final_recommendation", "Final Recommendation"),
]


class MessageBuffer:
    """Holds streaming state for the live display."""

    def __init__(self) -> None:
        self.agent_status: Dict[str, str] = {}
        for _team, agents in _TEAMS:
            for agent in agents:
                self.agent_status[agent] = "pending"
        self.current_report: str = ""
        self.current_section: str = ""
        self.report_count: int = 0
        self.node_updates: int = 0

    def mark_running(self, agent: str) -> None:
        if agent in self.agent_status:
            self.agent_status[agent] = "in_progress"

    def mark_done(self, agent: str) -> None:
        if agent in self.agent_status:
            self.agent_status[agent] = "completed"

    def set_report(self, title: str, content: str) -> None:
        self.current_section = title
        self.current_report = content
        self.report_count += 1

    def completed_count(self) -> int:
        return sum(1 for s in self.agent_status.values() if s == "completed")

    def total_count(self) -> int:
        return len(self.agent_status)


def create_layout() -> Layout:
    """Build the root layout tree (doc 10 §3)."""
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3),
    )
    return layout


def _progress_panel(buffer: MessageBuffer) -> Panel:
    table = Table(box=box.SIMPLE_HEAD, expand=True)
    table.add_column("Team", style="bold")
    table.add_column("Agent")
    table.add_column("Status", justify="center")
    for team, agents in _TEAMS:
        for agent in agents:
            status = buffer.agent_status.get(agent, "pending")
            if status == "pending":
                rendered = "[yellow]pending[/yellow]"
            elif status == "in_progress":
                rendered = "[bold cyan]running[/bold cyan]"
            elif status == "completed":
                rendered = "[green]completed[/green]"
            else:
                rendered = f"[red]{status}[/red]"
            table.add_row(team, agent, rendered)
        table.add_row("[dim]" + "─" * 12 + "[/dim]", "", "")
    return Panel(table, title="Progress", border_style="cyan")


def _analysis_panel(buffer: MessageBuffer) -> Panel:
    if buffer.current_report:
        body = Markdown(buffer.current_report)
        title = f"Latest: {buffer.current_section}" if buffer.current_section else "Analysis"
    else:
        body = Markdown("*Waiting for the first analyst report…*")
        title = "Analysis"
    return Panel(body, title=title, border_style="blue")


def _footer(buffer: MessageBuffer, start_time: float) -> Panel:
    elapsed = int(time.time() - start_time)
    mm, ss = divmod(elapsed, 60)
    parts = [
        f"Agents: {buffer.completed_count()}/{buffer.total_count()}",
        f"Reports: {buffer.report_count}",
        f"Updates: {buffer.node_updates}",
        f"⏱ {mm:02d}:{ss:02d}",
    ]
    return Panel(" | ".join(parts), border_style="green")


def _render(layout: Layout, buffer: MessageBuffer, start_time: float) -> None:
    layout["header"].update(
        Panel("[bold green]SportAgent — live analysis[/bold green]", border_style="green")
    )
    layout["main"].split_row(
        Layout(_progress_panel(buffer), name="progress", ratio=2),
        Layout(_analysis_panel(buffer), name="analysis", ratio=3),
    )
    layout["footer"].update(_footer(buffer, start_time))


def _apply_delta(buffer: MessageBuffer, node_name: str, delta: dict) -> None:
    """Update the buffer from one streamed node delta."""
    buffer.node_updates += 1
    mapping = _NODE_TO_AGENT.get(node_name)
    if mapping:
        _team, label = mapping
        buffer.mark_done(label)
    if isinstance(delta, dict):
        # Surface the most-recent report section produced by this node.
        for key, title in _REPORT_KEYS:
            content = delta.get(key)
            if isinstance(content, str) and content.strip():
                buffer.set_report(title, content)


def run_live(
    graph,
    query: str,
    sport: str,
    console: Optional[Console] = None,
    game_date: Optional[str] = None,
):
    """Stream a run through a rich.Live UI. Returns (final_state, rec).

    Falls back to a plain (non-live) drain of the stream on any Live error.
    """
    console = console or Console()
    buffer = MessageBuffer()
    start_time = time.time()
    final_state: dict = {}
    recommendation = ""

    try:
        from rich.live import Live

        layout = create_layout()
        with Live(layout, console=console, refresh_per_second=4, screen=False) as live:
            for node_name, payload in graph.analyze_stream(
                query, sport=sport, game_date=game_date
            ):
                if node_name == "__final__":
                    final_state, recommendation = payload
                    break
                # Mark the next agent running heuristically (the node that just
                # emitted is done; its successor will flip on its own delta).
                mapping = _NODE_TO_AGENT.get(node_name)
                if mapping:
                    buffer.mark_running(mapping[1])
                _apply_delta(buffer, node_name, payload if isinstance(payload, dict) else {})
                _render(layout, buffer, start_time)
                live.refresh()
        # Final render so the completed state is visible.
        _render(layout, buffer, start_time)
    except Exception:  # noqa: BLE001 — fall back to a plain drain
        for node_name, payload in graph.analyze_stream(
            query, sport=sport, game_date=game_date
        ):
            if node_name == "__final__":
                final_state, recommendation = payload
                break
    return final_state, recommendation
