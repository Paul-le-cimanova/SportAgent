"""SportAgent command-line interface (Typer + rich).

Commands:
- ``setup``    : interactive wizard that writes API keys to ``.env``.
- ``doctor``   : preflight report (presence + optional live ``--live`` pings).
- ``analyze``  : run the full pipeline for a Kalshi market / matchup query.

The package ``__init__`` loads ``.env`` at import, so every command sees the
user's keys regardless of how the console script is launched.
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sportagent import onboarding

app = typer.Typer(
    add_completion=False,
    invoke_without_command=True,
    help="SportAgent — winner-first NBA predictions on Kalshi sports markets.",
)
console = Console()


@app.callback()
def _main(ctx: typer.Context) -> None:
    """Launch the interactive game-picker wizard when no subcommand is given."""
    if ctx.invoked_subcommand is not None:
        return
    _run_wizard()


_STATUS_STYLE = {
    onboarding.STATUS_OK: "green",
    onboarding.STATUS_MISSING: "red",
    onboarding.STATUS_INVALID: "yellow",
    onboarding.STATUS_SKIPPED: "dim",
}


def _render_report(results) -> None:
    """Render a preflight report as a rich table."""
    table = Table(title="SportAgent environment check", show_lines=False)
    table.add_column("Key / check", no_wrap=True)
    table.add_column("Group")
    table.add_column("Req", justify="center")
    table.add_column("Status", justify="center")
    table.add_column("Hint", overflow="fold")
    for r in results:
        style = _STATUS_STYLE.get(r.status, "white")
        table.add_row(
            r.name,
            r.group,
            "✓" if r.required else "",
            f"[{style}]{r.status}[/{style}]",
            r.hint,
        )
    console.print(table)


@app.command()
def setup() -> None:
    """Interactively configure API keys and write them to ``.env``."""
    ok, message = onboarding.run_setup_wizard(console=console)
    if ok:
        console.print("\nRun [bold]sportagent doctor --live[/bold] to verify your keys.")
    else:
        console.print(f"[red]{message}[/red]")
        raise typer.Exit(code=1)


@app.command()
def doctor(
    live: bool = typer.Option(
        False, "--live", help="Run live API ping checks (uses a small amount of quota)."
    ),
) -> None:
    """Show a preflight report of required/optional keys (and optional pings)."""
    results = onboarding.check_environment(live=live)
    _render_report(results)
    missing = onboarding.required_missing(results)
    if missing:
        names = ", ".join(m.name for m in missing)
        console.print(
            Panel(
                f"Missing/invalid required keys: [red]{names}[/red]\n"
                "Run [bold]sportagent setup[/bold] to configure them.",
                title="Not ready",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)
    console.print("[green]All required keys present.[/green]")


def _preflight_or_exit() -> None:
    """Block analysis if required keys are missing; point to ``setup``."""
    results = onboarding.check_environment(live=False)
    missing = onboarding.required_missing(results)
    if missing:
        names = ", ".join(m.name for m in missing)
        console.print(
            Panel(
                f"Cannot analyze — missing/invalid required keys: [red]{names}[/red]\n"
                "Run [bold]sportagent setup[/bold] first, "
                "then [bold]sportagent doctor --live[/bold] to verify.",
                title="Setup required",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)


@app.command()
def analyze(
    query: str = typer.Argument(
        ..., help="Kalshi market ticker (exact) or a matchup like 'Spurs @ Knicks'."
    ),
    sport: str = typer.Option("nba", "--sport", help="Sport adapter key."),
    game_date: Optional[str] = typer.Option(
        None, "--game-date", help="Game date YYYY-MM-DD (picks the right-dated Kalshi market)."
    ),
    kalshi_env: Optional[str] = typer.Option(
        None, "--kalshi-env", help="Override Kalshi environment: demo|prod."
    ),
    deep_llm: Optional[str] = typer.Option(
        None, "--deep-llm", help="Override the deep-think model."
    ),
    quick_llm: Optional[str] = typer.Option(
        None, "--quick-llm", help="Override the quick-think model."
    ),
    live: bool = typer.Option(
        True, "--live/--no-live", help="Stream the run through the live UI."
    ),
    save: bool = typer.Option(
        True, "--save/--no-save", help="Save the report to ~/.sportagent/results/."
    ),
    debug: bool = typer.Option(False, "--debug", help="Verbose graph logging."),
) -> None:
    """Run the full pipeline for QUERY and print the final recommendation."""
    _preflight_or_exit()

    from sportagent.core.dataflows.config import get_config
    from sportagent.core.graph.sport_graph import SportAgentGraph

    config = get_config()
    if kalshi_env:
        config["kalshi_env"] = kalshi_env
    if deep_llm:
        config["deep_think_llm"] = deep_llm
    if quick_llm:
        config["quick_think_llm"] = quick_llm

    console.print(
        Panel(
            f"[bold]{query}[/bold]  ([cyan]{sport}[/cyan], kalshi={config.get('kalshi_env')})",
            title="SportAgent analyze",
            border_style="cyan",
        )
    )

    graph = SportAgentGraph(config=config, debug=debug)
    if live:
        state, recommendation = _run_with_live(graph, query, sport, game_date=game_date)
    else:
        state, recommendation = graph.analyze(query, sport=sport, game_date=game_date)

    _render_result(state, recommendation)
    if save:
        _save_and_report(state)


def _run_wizard() -> None:
    """Schedule-driven wizard → live-streamed analysis → saved report."""
    _preflight_or_exit()

    from sportagent import wizard

    result = wizard.run_game_wizard()
    if result is None:
        console.print("[yellow]No game selected — exiting.[/yellow]")
        raise typer.Exit(code=0)

    from sportagent.core.dataflows.config import get_config
    from sportagent.core.graph.sport_graph import SportAgentGraph

    config = get_config()
    config.update(result.config_overrides)

    console.print(
        Panel(
            f"[bold]{result.away} @ {result.home}[/bold]  "
            f"([cyan]{result.sport}[/cyan], {result.game_date}, "
            f"depth={result.research_depth})",
            title="SportAgent",
            border_style="cyan",
        )
    )

    graph = SportAgentGraph(config=config)
    state, recommendation = _run_with_live(
        graph, result.query, result.sport, game_date=result.game_date
    )
    _render_result(state, recommendation)
    _save_and_report(state)


def _run_with_live(graph, query: str, sport: str, game_date: Optional[str] = None):
    """Stream the run through the live UI, falling back to a blocking run."""
    try:
        from sportagent.live_ui import run_live

        return run_live(graph, query, sport, console=console, game_date=game_date)
    except Exception as exc:  # noqa: BLE001 — fall back to the blocking path
        console.print(f"[dim]Live UI unavailable ({exc}); running normally…[/dim]")
        return graph.analyze(query, sport=sport, game_date=game_date)


def _save_and_report(state: dict) -> None:
    """Save the completed report and print the path."""
    try:
        from sportagent.reporting import save_report

        ok, path = save_report(state)
        if ok:
            console.print(f"\n[green]Report saved to[/green] [bold]{path}[/bold]")
        else:
            console.print(f"[yellow]Could not save report: {path}[/yellow]")
    except Exception as exc:  # noqa: BLE001 — saving is best-effort
        console.print(f"[yellow]Could not save report: {exc}[/yellow]")


def _render_result(state: dict, recommendation: str) -> None:
    """Pretty-print the winner prediction first, then supporting detail."""
    # Winner-first headline (the MVP output).
    from sportagent.core.graph.signal_processing import parse_winner

    winner, win_prob = parse_winner(recommendation)
    if winner:
        loser_pct = (1.0 - win_prob) * 100
        console.print(
            Panel(
                f"🏀 [bold]{winner}[/bold] to win — "
                f"[bold green]{win_prob * 100:.0f}%[/bold green] "
                f"(opponent {loser_pct:.0f}%)",
                title="Prediction",
                border_style="green",
            )
        )

    verified = state.get("verified_odds", "")
    if verified:
        console.print(Panel(str(verified), title="Verified odds", border_style="blue"))

    thesis = state.get("investment_plan", "")
    if thesis:
        console.print(Panel(str(thesis), title="Edge thesis", border_style="magenta"))

    position = state.get("trader_position_plan", "")
    if position:
        console.print(Panel(str(position), title="Position proposal", border_style="yellow"))

    console.print(
        Panel(
            recommendation or "<no recommendation produced>",
            title="Full recommendation (prediction + betting view)",
            border_style="green",
        )
    )


if __name__ == "__main__":
    app()