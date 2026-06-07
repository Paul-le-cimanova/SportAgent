"""rich.Live streaming UI for a SportAgent run.

Renders a live layout while ``SportAgentGraph.analyze_stream`` runs:

  - Header     : welcome banner
  - Progress   : Team/Agent/Status table (pending -> running [spinner] -> done)
  - Analysis   : the active agent's live-streaming text / latest report section
  - Footer     : Agents / Reports / elapsed stats (ticks continuously)

Key behaviors that keep it from looking stuck:
  - One agent shows a spinner ``running`` at a time; status advances by the
    known pipeline order (fixes out-of-order completed/pending). Known-slow
    deep-model steps show a "may take a few min" hint; any step running long
    shows "still working…".
  - Steady refresh on a background thread so the spinner + clock animate even
    during multi-second gaps between graph chunks.
  - Logging is muted during the Live session and the alternate screen is used,
    so stray log lines never corrupt the layout.

Degrades gracefully: if rich.Live can't run (non-TTY / tiny terminal / error),
the caller's stream is drained plainly and a simple status line is printed.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.layout import Layout
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from rich import box

logger = logging.getLogger(__name__)

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

# Ordered teams + their agents for the progress panel (also the pipeline order).
_TEAMS: List[Tuple[str, List[str]]] = [
    ("Analyst Team", ["Odds Analyst", "Stats Analyst", "News-Injury Analyst", "Sentiment Analyst"]),
    ("Research Team", ["Bull Researcher", "Bear Researcher", "Research Manager"]),
    ("Trader", ["Trader"]),
    ("Risk Management", ["Aggressive", "Neutral", "Conservative"]),
    ("Decision", ["Decision Manager"]),
]

# Flattened pipeline order (for advancing running/done deterministically).
_PIPELINE: List[str] = [agent for _team, agents in _TEAMS for agent in agents]

# Agents that run the deep-think model and routinely take a few minutes. Shown
# with a reassuring sub-label so a long call reads as "working", not "stuck".
_DEEP_AGENTS = {"Research Manager", "Trader", "Decision Manager"}

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

# Noisy loggers to silence during a Live session (expected fail-open chatter).
_NOISY_LOGGERS = (
    "sportagent.sports.nba.stats",
    "sportagent.core.dataflows.reddit",
    "sportagent.core.dataflows.kalshi",
    "sportagent.core.dataflows.odds_api",
    "sportagent.core.dataflows.openweb_news",
    "sportagent.core.dataflows.odds_validator",
    "httpx",
)


class MessageBuffer:
    """Holds streaming state for the live display (thread-safe enough for UI)."""

    def __init__(self) -> None:
        self.agent_status: Dict[str, str] = {a: "pending" for a in _PIPELINE}
        self.current_agent: Optional[str] = None
        self.current_section: str = ""
        self.report_text: str = ""      # last completed section's full text
        self.report_count: int = 0
        self.node_updates: int = 0
        self.step_started: float = time.time()

    # --- status transitions ---

    def start_first(self) -> None:
        if _PIPELINE:
            self.agent_status[_PIPELINE[0]] = "running"
            self.current_agent = _PIPELINE[0]
            self.step_started = time.time()

    def advance_after(self, finished_label: str) -> None:
        """Mark ``finished_label`` done and set the next pipeline agent running.

        Earlier agents that are somehow still pending/running are marked done so
        the table never shows a gap. Handles debate loops gracefully (a repeated
        node just keeps the next one running).
        """
        if finished_label not in self.agent_status:
            return
        idx = _PIPELINE.index(finished_label)
        # Everything up to & including the finished agent is completed.
        for a in _PIPELINE[: idx + 1]:
            self.agent_status[a] = "completed"
        # The next not-yet-completed agent becomes running.
        nxt = next((a for a in _PIPELINE[idx + 1:] if self.agent_status[a] != "completed"), None)
        if nxt is not None:
            self.agent_status[nxt] = "running"
            self.current_agent = nxt
        else:
            self.current_agent = None
        self.step_started = time.time()

    def finish_all(self) -> None:
        for a in _PIPELINE:
            self.agent_status[a] = "completed"
        self.current_agent = None

    # --- content ---

    def set_report(self, title: str, content: str) -> None:
        self.current_section = title
        self.report_text = content
        self.report_count += 1

    # --- counters ---

    def completed_count(self) -> int:
        return sum(1 for s in self.agent_status.values() if s == "completed")

    def total_count(self) -> int:
        return len(self.agent_status)


def create_layout() -> Layout:
    """Build the root layout tree."""
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3),
    )
    layout["main"].split_row(
        Layout(name="progress", ratio=2),
        Layout(name="analysis", ratio=3),
    )
    return layout


_SPINNER = Spinner("dots", style="bold cyan")


def _progress_panel(buffer: MessageBuffer) -> Panel:
    table = Table(box=box.SIMPLE_HEAD, expand=True)
    table.add_column("Team", style="bold")
    table.add_column("Agent")
    table.add_column("Status", justify="left")
    for team, agents in _TEAMS:
        for agent in agents:
            status = buffer.agent_status.get(agent, "pending")
            if status == "pending":
                cell: object = Text("pending", style="yellow")
            elif status == "running":
                elapsed = int(time.time() - buffer.step_started)
                mm, ss = divmod(elapsed, 60)
                cell = Table.grid(padding=(0, 1))
                cell.add_row(_SPINNER, Text(f"running {mm:02d}:{ss:02d}", style="bold cyan"))
                # Reassure on known-slow deep-model steps, or any step that has
                # been running a while, so it never looks frozen.
                if agent in _DEEP_AGENTS:
                    cell.add_row("", Text("deep reasoning — may take a few min…", style="dim italic"))
                elif elapsed >= 45:
                    cell.add_row("", Text("still working…", style="dim italic"))
            elif status == "completed":
                cell = Text("✓ completed", style="green")
            else:
                cell = Text(status, style="red")
            table.add_row(team, agent, cell)
        table.add_row(Text("─" * 12, style="dim"), "", "")
    return Panel(table, title="Progress", border_style="cyan")


def _analysis_panel(buffer: MessageBuffer) -> Panel:
    if buffer.report_text:
        body: object = Markdown(buffer.report_text)
        title = f"Latest: {buffer.current_section}" if buffer.current_section else "Analysis"
    else:
        who = buffer.current_agent or "the first analyst"
        body = Text(f"Working — {who} is analyzing…", style="italic dim")
        title = "Analysis"
    return Panel(body, title=title, border_style="blue")


def _footer(buffer: MessageBuffer, start_time: float) -> Panel:
    elapsed = int(time.time() - start_time)
    mm, ss = divmod(elapsed, 60)
    active = buffer.current_agent or "—"
    parts = [
        f"Agents: {buffer.completed_count()}/{buffer.total_count()}",
        f"Active: {active}",
        f"Reports: {buffer.report_count}",
        f"Updates: {buffer.node_updates}",
        f"⏱ {mm:02d}:{ss:02d}",
    ]
    return Panel(" | ".join(parts), border_style="green")


def _render(layout: Layout, buffer: MessageBuffer, start_time: float) -> None:
    layout["header"].update(
        Panel("[bold green]SportAgent — live analysis[/bold green]", border_style="green")
    )
    layout["progress"].update(_progress_panel(buffer))
    layout["analysis"].update(_analysis_panel(buffer))
    layout["footer"].update(_footer(buffer, start_time))


def _apply_update(buffer: MessageBuffer, node_name: str, delta: dict) -> None:
    """Handle one node-level state update."""
    buffer.node_updates += 1
    mapping = _NODE_TO_AGENT.get(node_name)
    if mapping:
        # Log how long the just-finished step took so a slow agent is visible
        # in the run log even when the live UI is muted.
        elapsed = time.time() - buffer.step_started
        logger.info("Node %r finished in %.1fs", node_name, elapsed)
        buffer.advance_after(mapping[1])
    if isinstance(delta, dict):
        for key, title in _REPORT_KEYS:
            content = delta.get(key)
            if isinstance(content, str) and content.strip():
                buffer.set_report(title, content)


class _MuteNoisyLogs:
    """Context manager: raise noisy data-layer loggers to ERROR during Live."""

    def __enter__(self):
        self._saved = {}
        for name in _NOISY_LOGGERS:
            lg = logging.getLogger(name)
            self._saved[name] = lg.level
            lg.setLevel(logging.ERROR)
        return self

    def __exit__(self, *exc):
        for name, level in self._saved.items():
            logging.getLogger(name).setLevel(level)
        return False


def _stream_into(buffer: MessageBuffer, graph, query, sport, game_date, result_box):
    """Background worker: drain analyze_stream into the buffer. Stores result."""
    run_started = time.time()
    logger.info("Live run started: query=%r sport=%r date=%r", query, sport, game_date)
    try:
        for node_name, payload in graph.analyze_stream(query, sport=sport, game_date=game_date):
            if node_name == "__final__":
                result_box["state"], result_box["rec"] = payload
                buffer.finish_all()
                logger.info("Live run finished in %.1fs total", time.time() - run_started)
                break
            _apply_update(buffer, node_name, payload if isinstance(payload, dict) else {})
    except Exception as exc:  # noqa: BLE001 — surface, don't crash the UI thread
        logger.exception("Live stream worker failed")
        result_box["error"] = exc
    finally:
        result_box["done"] = True


def _can_use_live(console: Console) -> bool:
    """True only when the terminal can host a full-screen rich.Live layout."""
    try:
        if not console.is_terminal:
            return False
        size = console.size
        return size.width >= 80 and size.height >= 20
    except Exception:  # noqa: BLE001
        return False


def run_live(
    graph,
    query: str,
    sport: str,
    console: Optional[Console] = None,
    game_date: Optional[str] = None,
):
    """Stream a run through a rich.Live UI. Returns (final_state, rec).

    Drives the graph stream on a background thread and refreshes the layout on a
    steady ~8 fps loop so the spinner + clock animate during long LLM calls.
    Falls back to a plain drain (with a simple status line) when the terminal
    can't host the layout or Live errors out.
    """
    console = console or Console()
    buffer = MessageBuffer()
    start_time = time.time()
    result_box: dict = {"state": {}, "rec": "", "done": False, "error": None}

    if not _can_use_live(console):
        return _run_plain(graph, query, sport, console, game_date)

    try:
        from rich.live import Live

        buffer.start_first()
        layout = create_layout()
        worker = threading.Thread(
            target=_stream_into,
            args=(buffer, graph, query, sport, game_date, result_box),
            daemon=True,
        )
        with _MuteNoisyLogs():
            with Live(
                layout,
                console=console,
                refresh_per_second=8,
                screen=True,            # alternate screen — clean repaint in VS Code
                redirect_stdout=True,   # capture stray prints instead of corrupting
                redirect_stderr=True,
            ) as live:
                worker.start()
                # Steady refresh loop — keeps spinner + clock moving between chunks.
                while not result_box["done"]:
                    _render(layout, buffer, start_time)
                    live.refresh()
                    time.sleep(0.125)
                _render(layout, buffer, start_time)
                live.refresh()
        worker.join(timeout=1.0)
    except Exception:  # noqa: BLE001 — fall back to a plain drain
        if not result_box["done"]:
            return _run_plain(graph, query, sport, console, game_date)

    if result_box.get("error") is not None:
        console.print(f"[yellow]Live run error: {result_box['error']}[/yellow]")
    return result_box.get("state", {}), result_box.get("rec", "")


def _run_plain(graph, query: str, sport: str, console: Console, game_date: Optional[str]):
    """Plain (non-Live) drain with a simple per-agent status line. Returns (state, rec)."""
    final_state: dict = {}
    recommendation = ""
    console.print("[dim]Running analysis (simple mode)…[/dim]")
    with _MuteNoisyLogs():
        for node_name, payload in graph.analyze_stream(query, sport=sport, game_date=game_date):
            if node_name == "__final__":
                final_state, recommendation = payload
                break
            mapping = _NODE_TO_AGENT.get(node_name)
            if mapping:
                console.print(f"  [green]✓[/green] {mapping[1]}")
    return final_state, recommendation