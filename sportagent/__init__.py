"""SportAgent — multi-agent LLM framework for Kalshi sports markets.

Loads ``.env`` at import so config and data-layer clients see the user's
keys regardless of entry point (CLI console script, ``python main.py``, or
programmatic import).
"""

try:
    from dotenv import find_dotenv, load_dotenv

    # Walk up from the current working directory so an installed console
    # script picks up the project's .env rather than stepping up from
    # site-packages. override=False never clobbers an already-exported var.
    load_dotenv(find_dotenv(usecwd=True), override=False)
except ImportError:  # python-dotenv optional at runtime
    pass

__version__ = "0.1.0"