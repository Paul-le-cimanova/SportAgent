"""SportAgent command-line interface (Typer + rich).

Commands:
- ``setup``    : interactive wizard that writes API keys to ``.env``.
- ``doctor``   : preflight report (presence + optional live ``--live`` pings).
- ``analyze``  : run the full pipeline for a Kalshi market / matchup query.
- ``update``   : check PyPI for a newer version and upgrade in place.

The package ``__init__`` loads ``.env`` at import, so every command sees the
user's keys regardless of how the console script is launched.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from sportagent import onboarding
from sportagent.run_logging import setup_run_logging

app = typer.Typer(
    add_completion=False,
    invoke_without_command=True,
    help="SportAgent — winner-first NBA predictions on Kalshi sports markets.",
)
console = Console()


@app.callback()
def _main(ctx: typer.Context) -> None:
    """Launch the interactive game-picker wizard when no subcommand is given."""
    # Never run (or print from) the update check during `update` itself.
    if ctx.invoked_subcommand != "update":
        _check_for_update_background()
    if ctx.invoked_subcommand is not None:
        return
    _run_wizard()


def _installed_version() -> str:
    """Return the installed sportagent version (best-effort)."""
    try:
        from importlib.metadata import version as installed_version

        return installed_version("sportagent")
    except Exception:  # noqa: BLE001 — e.g. running from a source checkout
        try:
            import sportagent

            return getattr(sportagent, "__version__", "0.0.0")
        except Exception:  # noqa: BLE001
            return "0.0.0"


def _get_latest_pypi_version(package: str) -> Optional[str]:
    """Fetch the latest version from the PyPI JSON API (None on any error)."""
    import urllib.request

    try:
        url = f"https://pypi.org/pypi/{package}/json"
        with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310 — fixed https URL
            data = json.loads(resp.read())
            return data["info"]["version"]
    except Exception:  # noqa: BLE001 — fail open, never block the CLI
        return None


_UPDATE_CACHE_PATH = Path.home() / ".sportagent" / "update_check.json"
_UPDATE_CACHE_TTL_SECONDS = 86400  # 24h


def _check_for_update_background() -> None:
    """Non-blocking check for a newer version; one-line notice if outdated.

    Results are cached at ``~/.sportagent/update_check.json`` for 24h so the
    network is hit at most once a day. Any failure is silent — the check must
    never interfere with a run.
    """

    def _notify(latest: str) -> None:
        console.print(
            f"[dim]SportAgent v{latest} available — "
            f"run `sportagent update` to upgrade[/dim]"
        )

    def _check() -> None:
        try:
            # Fresh cache → use it, skip the network.
            if _UPDATE_CACHE_PATH.exists():
                try:
                    data = json.loads(_UPDATE_CACHE_PATH.read_text())
                    if time.time() - data.get("ts", 0) < _UPDATE_CACHE_TTL_SECONDS:
                        if data.get("newer") and data.get("latest"):
                            _notify(data["latest"])
                        return
                except Exception:  # noqa: BLE001 — corrupt cache → re-check
                    pass

            current = _installed_version()
            latest = _get_latest_pypi_version("sportagent")
            if latest is None:
                return
            newer = latest != current
            _UPDATE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            _UPDATE_CACHE_PATH.write_text(
                json.dumps(
                    {"ts": time.time(), "current": current, "latest": latest, "newer": newer}
                )
            )
            if newer:
                _notify(latest)
        except Exception:  # noqa: BLE001 — never let the check crash anything
            pass

    threading.Thread(target=_check, daemon=True).start()


@app.command()
def update() -> None:
    """Check for updates and install the latest version from PyPI."""
    import subprocess
    import sys

    current = _installed_version()
    console.print(f"Current version: [bold]{current}[/bold]")
    console.print("Checking for updates…")

    latest = _get_latest_pypi_version("sportagent")
    if latest is None:
        console.print("[yellow]Could not check PyPI for updates.[/yellow]")
        raise typer.Exit(code=1)
    if latest == current:
        console.print(f"[green]Already up to date (v{current}).[/green]")
        return

    console.print(f"New version available: [bold green]v{latest}[/bold green]")
    console.print("Upgrading…")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "sportagent"]
        )
    except subprocess.CalledProcessError as exc:
        console.print(f"[red]Upgrade failed (pip exited with {exc.returncode}).[/red]")
        raise typer.Exit(code=1)

    # Invalidate the 24h cache so the startup notice disappears immediately.
    try:
        _UPDATE_CACHE_PATH.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001 — cache cleanup is best-effort
        pass
    console.print(f"[green]Updated to v{latest}![/green]")


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

    log_path = setup_run_logging(matchup=query, verbose=debug)

    graph = SportAgentGraph(config=config, debug=debug)
    if live:
        state, recommendation = _run_with_live(graph, query, sport, game_date=game_date)
    else:
        state, recommendation = graph.analyze(query, sport=sport, game_date=game_date)

    _render_result(state, recommendation)
    if save:
        _save_and_report(state)
    _report_log_path(state, log_path)


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

    log_path = setup_run_logging(matchup=result.query)

    graph = SportAgentGraph(config=config)
    state, recommendation = _run_with_live(
        graph, result.query, result.sport, game_date=result.game_date
    )
    _render_result(state, recommendation)
    _save_and_report(state)
    _report_log_path(state, log_path)


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
    """Pretty-print the winner prediction first, then supporting detail.

    Content fields are markdown (headers/bold/bullets), so they are rendered
    with ``rich.markdown.Markdown`` rather than printed as raw text.
    """
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
        console.print(Panel(Markdown(str(verified)), title="Verified odds", border_style="blue"))

    thesis = state.get("investment_plan", "")
    if thesis:
        console.print(Panel(Markdown(str(thesis)), title="Edge thesis", border_style="magenta"))

    position = state.get("trader_position_plan", "")
    if position:
        console.print(Panel(Markdown(str(position)), title="Position proposal", border_style="yellow"))

    rec = recommendation or "<no recommendation produced>"
    if rec.startswith("<error"):
        console.print(Panel(rec, title="Run failed", border_style="red"))
    else:
        console.print(
            Panel(
                Markdown(rec),
                title="Full recommendation (prediction + betting view)",
                border_style="green",
            )
        )


def _report_log_path(state: dict, log_path) -> None:
    """Tell the user where the run log is — prominently when the run errored."""
    if log_path is None:
        return
    if state.get("error") or str(state.get("final_recommendation", "")).startswith("<error"):
        console.print(
            Panel(
                f"This run hit an error. Full traceback + logs:\n[bold]{log_path}[/bold]",
                title="Run log",
                border_style="red",
            )
        )
    else:
        console.print(f"[dim]Run log: {log_path}[/dim]")


if __name__ == "__main__":
    app()
