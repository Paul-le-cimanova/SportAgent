"""Vendor-abstracted data layer for SportAgent.

Importing this package eagerly imports every vendor module so each one runs its
``register_vendor_method(...)`` calls at import time. Without this, modules like
``odds_api``/``reddit``/``openweb_news`` would only register if some other code
happened to import them first, leaving their routed tools unregistered (the
"data blackout" symptom). Each import is best-effort so a single bad module
never breaks the whole data layer.
"""

import logging as _logging

_logger = _logging.getLogger(__name__)

# Order: register everything. kalshi_auth has no registrations but is a dep.
_VENDOR_MODULES = (
    "sportagent.core.dataflows.kalshi",
    "sportagent.core.dataflows.odds_api",
    "sportagent.core.dataflows.reddit",
    "sportagent.core.dataflows.openweb_news",
    # NBA stats tools (balldontlie + espn) live under sports/, import for reg.
    "sportagent.sports.nba.stats",
)

for _mod in _VENDOR_MODULES:
    try:
        __import__(_mod)
    except Exception as _exc:  # noqa: BLE001 — never break import on one vendor
        _logger.warning("dataflows: failed to import vendor module %s: %s", _mod, _exc)
