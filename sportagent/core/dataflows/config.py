"""Shared runtime config with deep-copy isolation.

``get_config()`` returns a deep copy so callers never mutate the shared dict.
``set_config()`` merges dict-valued keys one level deep so a partial update
(e.g. one ``data_vendors`` entry) preserves sibling defaults.
"""

from copy import deepcopy
from typing import Dict, Optional

from sportagent.default_config import DEFAULT_CONFIG

_config: Optional[Dict] = None


def initialize_config() -> None:
    """Initialize the runtime config from DEFAULT_CONFIG if not already set."""
    global _config
    if _config is None:
        _config = deepcopy(DEFAULT_CONFIG)


def set_config(config: Dict) -> None:
    """Update the runtime config.

    Dict-valued keys (e.g. ``data_vendors``) are merged one level deep so a
    partial update keeps the other nested keys from the default; scalar keys
    are replaced.
    """
    global _config
    initialize_config()
    incoming = deepcopy(config)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(_config.get(key), dict):
            _config[key].update(value)
        else:
            _config[key] = value


def get_config() -> Dict:
    """Return a deep copy of the current runtime config."""
    if _config is None:
        initialize_config()
    return deepcopy(_config)


initialize_config()