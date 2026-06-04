<div align="center">

# SportAgent: Multi-Agent LLM Sports Prediction Framework

</div>

<div align="center">

⚡ [Installation & CLI](#installation-and-cli) | 📦 [Package Usage](#sportagent-package) | 🧠 [Persistence](#persistence-and-memory) | 🔁 [Reproducibility](#reproducibility) | 🤝 [Contributing](#contributing)

</div>

---

## News
- **[2026-06] SportAgent v0.1.1** — Kalshi data-correctness fixes: market prices now read the live `*_dollars` fields, and game-winner markets resolve to the **exact game date** you select (no more stale/expired tickers). Default Kalshi environment switched to **prod** (read-only) for real market prices.
- **[2026-06] SportAgent v0.1.0** — First public release. Winner-first NBA predictions, a schedule-driven game-picker wizard, a live streaming UI, and saved Markdown reports.

> ⚠️ **SportAgent is designed for research and educational purposes.** Predictions are probabilistic and depend on the chosen models, model temperature, data quality, and other non-deterministic factors. v1 is **read-only** — it never places orders. **This is not financial, investment, or betting advice.**

## SportAgent Framework

SportAgent is a multi-agent prediction framework that mirrors the dynamics of a real sports analysis desk. By deploying specialized LLM-powered agents — from an odds analyst, a stats analyst, a news/injury analyst, and a sentiment expert, to a bull/bear research debate, a trader, and a risk-management committee — the platform collaboratively evaluates a game and produces an explainable, **winner-first** prediction:

> 🏀 **PREDICTION: New York Knicks win — 64%** *(San Antonio Spurs 36%)*

The system cross-checks the [Kalshi](https://kalshi.com) sports-market contract price (the market-implied probability) against a de-vigged sportsbook consensus, treats that verified snapshot as the source of truth for exact prices, and then reasons about the **edge** between the market and its own estimate. All quantitative math (implied probability, vig removal, edge, Kelly sizing, Brier calibration) is **deterministic** — never left to the language model.

The framework is **sport-agnostic** at its core (`core/`), with each sport added as a self-contained adapter under `sports/<sport>/`. NBA ships first.

Our framework decomposes the prediction task into specialized roles.

### Analyst Team
- **Odds Analyst** — Reads the verified-odds snapshot (Kalshi contract price vs. de-vigged sportsbook consensus), flags any mispricing, and establishes the market-implied probability.
- **Stats Analyst** — Evaluates team strength: season record, recent form, head-to-head history, and rest / back-to-back status, weighing the factors a desk would price in.
- **News / Injury Analyst** — Monitors injury reports, lineup changes, and late-breaking availability news that can swing a game's win probability.
- **Sentiment Analyst** — Aggregates public and community chatter (Reddit / sportsbook discussion) into a single sentiment read, watching for contrarian signals where heavy public money on one side flags value on the other.

### Research Team
- Comprises a **Bull** and a **Bear** researcher who critically assess the analysts' findings. Through a structured debate, they weigh the case for and against the favored team, and a **Research Manager** (deep-think model) synthesizes the debate into a committed probability estimate.

### Trader
- Converts the research thesis into a concrete position, using the deterministic probability/Kelly helpers to size a (hypothetical) stake against the verified market price.

### Risk Management and Decision Manager
- A risk committee of **Aggressive**, **Neutral**, and **Conservative** voices debates the position from every angle.
- The **Decision Manager** (deep-think model) renders the final, winner-first call: which team wins, the win probability, a confidence level, and plain-language reasoning — with the betting view (BUY YES / BUY NO / HOLD, edge, suggested stake) retained as an optional secondary section.

## Installation and CLI

### Installation

Clone SportAgent:
```bash
git clone https://github.com/Paul-le-cimanova/SportAgent.git
cd SportAgent
```

Create a virtual environment in any environment manager you like:
```bash
conda create -n sportagent python=3.13
conda activate sportagent
```

Install the package and its dependencies (this also installs the `sportagent` command):
```bash
pip install -e .
```

### Configure your keys

```bash
sportagent setup          # interactive wizard → writes .env
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

Keys live in a gitignored `.env`. **Never commit your `.env` or your Kalshi `.pem`.** Kalshi defaults to **prod** (`KALSHI_ENV=prod`) because v1 is read-only and prod gives real market prices; set `KALSHI_ENV=demo` for the sandbox.

Alternatively, copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

### CLI Usage

Launch the interactive wizard:
```bash
sportagent               # bare command → game-picker wizard
python -m sportagent.cli # alternative: run directly from source
```

The wizard is **schedule-driven** — you never type team names. You'll pick a sport, then a date (Today / Tomorrow / Upcoming 7 days / a specific future date), then a real game from that day's schedule, then research depth and models. SportAgent then streams the multi-agent run live and saves a full report.

For power users, the one-liner skips the wizard:
```bash
sportagent analyze "Knicks @ Spurs" --game-date 2026-06-05
```

Flags: `--game-date YYYY-MM-DD`, `--sport`, `--kalshi-env demo|prod`, `--deep-llm`, `--quick-llm`, `--live/--no-live`, `--save/--no-save`.

### Reports

Each run is saved to `~/.sportagent/results/<matchup>/<date>/`:
- `complete_report.md` — the winner headline plus every section
- per-section Markdown (odds / stats / news / sentiment / research / trader / decision)
- `full_state.json` — the raw final pipeline state

## SportAgent Package

### Implementation Details

SportAgent is built on LangGraph for flexibility and modularity. A `StateGraph` runs the analyst → research → trader → risk → decision pipeline over a shared game state. Two LLM tiers are used: a **deep** model for the Research and Decision Managers, and a **quick** model for everyone else. It currently supports **Anthropic (Claude)** and **OpenAI (GPT)** providers.

Every data fetcher **fails open** — a dead or rate-limited source returns a clear placeholder string rather than crashing the run.

### Python Usage

To use SportAgent inside your code, import the package and initialize a `SportAgentGraph()`. The `.analyze()` method returns `(final_state, recommendation)`:

```python
from sportagent.core.graph.sport_graph import SportAgentGraph
from sportagent.default_config import DEFAULT_CONFIG

graph = SportAgentGraph(config=DEFAULT_CONFIG.copy(), debug=True)

state, recommendation = graph.analyze("Knicks @ Spurs", game_date="2026-06-05")
print(recommendation)
```

You can adjust the default configuration to set your own models, debate depth, and more:

```python
from sportagent.core.graph.sport_graph import SportAgentGraph
from sportagent.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "anthropic"                       # "anthropic" or "openai"
config["deep_think_llm"] = "claude-opus-4-8"               # managers
config["quick_think_llm"] = "claude-haiku-4-5-20251001"    # analysts / researchers / trader / risk
config["max_debate_rounds"] = 2
config["max_risk_rounds"] = 2

graph = SportAgentGraph(config=config)
state, recommendation = graph.analyze("Knicks @ Spurs", game_date="2026-06-05")
print(recommendation)
```

See `sportagent/default_config.py` for all configuration options.

## Persistence and Memory

SportAgent keeps an append-only **decision log** at `~/.sportagent/memory/sport_memory.md`. Each completed run records its prediction. Before a later run, SportAgent resolves any prior settled games (fetching the final Kalshi result), scores its earlier call with a Brier calibration, writes a short reflection, and injects the most recent same-matchup lessons plus a few cross-game lessons into the Decision Manager's prompt — so each analysis carries forward what worked and what didn't.

Override the path with `SPORTAGENT_MEMORY_LOG_PATH`.

## Reproducibility

SportAgent is LLM-driven, so two runs of the same game can differ. This is expected for a tool built on language models, not a defect, and the variation comes from a few distinct sources.

Language-model sampling is non-deterministic: even at a fixed temperature, providers do not guarantee byte-identical output, and reasoning models vary the most. Live data also moves — injury news, sentiment, and odds shift as a game approaches, so a run today sees different inputs than a run yesterday.

What does **not** vary: the game identity is resolved deterministically before any agent runs, and a **verified-odds snapshot** grounds every exact price claim, so the analysts cannot fabricate a different matchup or invent prices. All quantitative math is deterministic.

To reduce variation, lower the sampling temperature (set `temperature` in your config or `SPORTAGENT_TEMPERATURE` in `.env`) and pair it with a non-reasoning model.

## Multi-sport

The core is sport-agnostic; sports are added as adapters under `sportagent/sports/<sport>/`. NBA ships first. NFL / MLB / soccer are scaffolded — soccer needs 3-way win/draw/loss handling, already modeled in `MarketRef.outcome_structure`.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Contributing

Contributions are welcome — bug fixes, documentation, new sport adapters, and feature ideas.

## License

[MIT](LICENSE)