# SportAgent

**SportAgent** is a multi-agent LLM application that produces explainable, **winner-first** predictions for NBA games — *"🏀 PREDICTION: New York Knicks win — 64%"* — by debating real data the way a desk of analysts would. It cross-checks the [Kalshi](https://kalshi.com) sports market price against a de-vigged sportsbook consensus, then runs a team of specialized agents (odds, stats, news/injury, sentiment, a bull/bear research debate, and a risk committee) to estimate the true win probability.

It is inspired by — but does **not** import from — [TradingAgents](https://github.com/TauricResearch/TradingAgents). SportAgent is a brand-new, sport-agnostic codebase (`core/` + per-sport `sports/<sport>/` adapters), shipping with an NBA adapter first.

> ⚠️ **Analysis only. Not financial advice.** v1 is read-only — it never places orders. Predictions are probabilistic and for informational/educational use. Do your own research.

---

## What it does

A LangGraph pipeline of agents runs in sequence:

**Analyst Team** (Odds → Stats → News/Injury → Sentiment) → **Research Team** (Bull ↔ Bear debate → Research Manager) → **Trader** → **Risk Management** (Aggressive / Neutral / Conservative) → **Decision Manager**.

- The **primary output** is a game-winner prediction: which team wins, a win probability, a confidence level, and plain-language reasoning.
- A **secondary "Betting view"** is still computed (Kalshi BUY YES / BUY NO / HOLD, edge vs. market, Kelly-sized stake) but it is not the headline.
- All quantitative math (implied probability, de-vig, edge, Kelly, Brier) is **deterministic**, never left to the LLM.
- Every data fetcher **fails open** — a dead source yields a clear placeholder, never a crash.

## Install

Requires Python 3.10+.

```bash
git clone https://github.com/<your-username>/SportAgent.git
cd SportAgent
pip install -e .
```

This installs the `sportagent` console command.

## Configure your keys

```bash
sportagent setup        # interactive wizard → writes .env
sportagent doctor --live  # verify keys + live API pings
```

You'll need:

| Key | Required | What it's for | Get it |
| --- | --- | --- | --- |
| `ANTHROPIC_API_KEY` | ✅ | LLM (default: Claude) | https://console.anthropic.com |
| `KALSHI_ACCESS_KEY_ID` + `KALSHI_PRIVATE_KEY_PATH` | ✅ | Kalshi market prices (read-only) | https://kalshi.com/account/profile |
| `THE_ODDS_API_KEY` | ✅ | Sportsbook consensus odds (free tier) | https://the-odds-api.com |
| `BALLDONTLIE_API_KEY` | ✅ | NBA team stats / schedule | https://app.balldontlie.io |
| `OPENWEB_NINJA_API_KEY` | optional | Injury / lineup news | https://www.openwebninja.com |
| `OPENAI_API_KEY` | optional | If you prefer GPT models | https://platform.openai.com |

Keys live in a gitignored `.env`. **Never commit your `.env` or your Kalshi `.pem`.**

Kalshi defaults to **prod** (`KALSHI_ENV=prod`) because v1 is read-only and prod gives real market prices. Set `KALSHI_ENV=demo` to use the sandbox.

## Usage

### Interactive wizard (recommended)

```bash
sportagent
```

Bare `sportagent` launches a schedule-driven wizard: **pick sport → pick a date (today / tomorrow / upcoming 7 days / a future date) → pick a real game from the schedule → research depth → models.** You never type team names. It then streams the multi-agent run live and saves a full report.

### Power-user one-liner

```bash
sportagent analyze "Knicks @ Spurs" --game-date 2026-06-05
```

Flags: `--game-date YYYY-MM-DD`, `--sport`, `--kalshi-env demo|prod`, `--deep-llm`, `--quick-llm`, `--live/--no-live`, `--save/--no-save`.

## Reports

Each run is saved to `~/.sportagent/results/<matchup>/<date>/`:

- `complete_report.md` — winner headline + every section
- per-section markdown (odds / stats / news / sentiment / research / trader / decision)
- `full_state.json` — the raw final pipeline state

## How it works (one paragraph)

A LangGraph `StateGraph` runs the analyst → research → trader → risk → decision pipeline. The Kalshi contract price is the market-implied probability; the system estimates the true probability and reasons about the **edge**. Game identity and a **verified-odds snapshot** (Kalshi price cross-checked against the de-vigged sportsbook consensus) are treated as the source of truth for exact prices, guarding against hallucination. Two LLM tiers are used: a deep model for the Research and Decision Managers, a quick model for everyone else. After a game settles, a Brier-calibration reflection is logged and fed into future runs.

## Multi-sport

The core is sport-agnostic; sports are added as adapters under `sportagent/sports/<sport>/`. NBA ships first. NFL / MLB / soccer are scaffolded (soccer needs 3-way win/draw/loss handling, already modeled in `MarketRef.outcome_structure`).

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

[MIT](LICENSE)