# Changelog

All notable changes to SportAgent are documented here. This project follows
[Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`. While on `0.x`,
the project is pre-1.0 and still evolving — minor versions add features, patch
versions fix bugs.

## [0.1.3] — 2026-06-08

### Fixed
- **Trader probability-parse bug** — the Trader silently fell back to a 50%
  estimate whenever the Research Manager rendered its edge thesis as
  `0.610 (61.0%)` (a decimal immediately followed by a parenthesized percent).
  The old regex required a trailing `%` right after the first number and so
  never matched the canonical decimal, corrupting every downstream edge/Kelly
  calculation and producing contradictory runs (Manager 61% → Trader 50% → a
  flipped BUY NO). The parser now reads the percentage echo or the canonical
  decimal and only defaults to 0.5 when neither is present.
- **Four-Factors small-sample bug** — `/stats` returns one row *per player per
  game* (~30 rows/game), so a single 100-row page only covered ~3 games. The
  aggregation now follows `next_cursor` pagination so the Four Factors are
  computed over the full requested window (e.g. 20 games).

### Added
- **Real quantitative Stats Analyst tools** (the stockstats-equivalent the
  TradingAgents-style debate assumes):
  - `get_four_factors(team)` — season eFG%, **turnover rate**, offensive-rebound
    rate, and **free-throw rate** (plus PPG / FGA), the box-score rates that
    actually predict NBA outcomes. Turnover rate and free-throw rate are where
    close games are won.
  - `get_elo_winprob(home, away)` — a **real, deterministic** Elo win
    probability (home-court adjustment, margin-of-victory multiplier, playoff
    K-factor) the Research Manager can anchor to instead of inventing an "Elo
    model" in prose.
- **NBA key-factors prompt** now instructs the analyst to call the Elo prior and
  Four Factors first, cite them explicitly, never claim to have run a model it
  did not call as a tool, and keep single-game probabilities honest (high
  variance — rarely above ~65% without a major injury).
- Tests for the probability-parse fix, Elo math, and Four-Factors pagination
  (`tests/test_stats_signals.py`).

## [0.1.2] — 2026-06-07

### Fixed
- **Live-run crash** (`expected str instance, list found`) that aborted runs
  mid-pipeline — reverted the analyst stream to node-level updates
  (`stream_mode="updates"`) so Anthropic message content is no longer delivered
  as block-lists into a string join.
- **Final report rendering** — verified-odds, edge-thesis, position-proposal and
  the full recommendation panels now render Markdown (headers/bold/bullets)
  instead of printing raw `**`/`##`/`###`.
- Removed the internal "Proceed with…" placeholder that leaked into the UI.

### Added
- **Per-run logging** to `~/.sportagent/logs/run-<matchup>-<timestamp>.log`
  (full DEBUG with tracebacks); the CLI prints the log path, prominently on
  failure, and `analyze --debug` streams logs to the console.
- **Per-agent + total run timings** in the run log so a slow step is easy to spot.
- **"Working, not stuck" UI hints** — deep-model steps (Research Manager,
  Trader, Decision Manager) show a "deep reasoning — may take a few min…" label,
  and any step running ≥45s shows "still working…", under the live spinner/clock.

## [0.1.1] — 2026-06-04

### Fixed
- **Kalshi market prices** now read the live `*_dollars` string fields
  (`last_price_dollars`, `yes_bid_dollars`, `yes_ask_dollars`) instead of the
  legacy integer-cents fields, so contract prices resolve correctly instead of
  falling back to the sportsbook consensus.
- **Date-aware market resolution** — the game date you select now flows
  end-to-end and SportAgent resolves the Kalshi market for that exact date
  (no more matching to a stale/expired ticker), and reports are stamped with the
  date you actually picked.

### Changed
- Default Kalshi environment switched from `demo` to **`prod`** (v1 is
  read-only, so prod gives real market prices). Set `KALSHI_ENV=demo` for the
  sandbox.
- Expanded the README into a full framework-style overview (agent team
  breakdown, install/CLI, package usage, persistence, reproducibility).

## [0.1.0] — 2026-06-04

### Added
- First public release.
- **Winner-first NBA predictions** — the headline output is which team wins,
  with a win probability, confidence, and plain-language reasoning. The betting
  view (BUY YES / BUY NO / HOLD, edge, Kelly stake) is retained as an optional
  secondary section.
- **Schedule-driven game-picker wizard** — `sportagent` launches an interactive
  flow: pick sport → date → a real game from the schedule → research depth →
  models. No typing team names.
- **Live streaming UI** showing each agent flip pending → running → completed.
- **Saved Markdown reports** to `~/.sportagent/results/<matchup>/<date>/`.
- Multi-agent pipeline: Odds / Stats / News-Injury / Sentiment analysts →
  Bull/Bear research debate → Research Manager → Trader → Aggressive / Neutral /
  Conservative risk debate → Decision Manager.
- Deterministic quantitative core (implied probability, vig removal, edge, Kelly
  sizing, Brier calibration) and a verified-odds snapshot for anti-hallucination.
- Append-only decision log with post-settlement Brier reflection fed into future
  runs.

[0.1.3]: https://github.com/Paul-le-cimanova/SportAgent/releases/tag/v0.1.3
[0.1.2]: https://github.com/Paul-le-cimanova/SportAgent/releases/tag/v0.1.2
[0.1.1]: https://github.com/Paul-le-cimanova/SportAgent/releases/tag/v0.1.1
[0.1.0]: https://github.com/Paul-le-cimanova/SportAgent/releases/tag/v0.1.0