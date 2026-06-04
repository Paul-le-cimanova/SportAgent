"""Kalshi RSA request signing.

Kalshi authenticates each request with an RSA key pair (not a bearer token).
Per request, three headers are sent:

- ``KALSHI-ACCESS-KEY``        : the Access Key ID
- ``KALSHI-ACCESS-TIMESTAMP``  : current epoch milliseconds (as a string)
- ``KALSHI-ACCESS-SIGNATURE``  : base64 RSA-PSS-SHA256 signature of the string
                                 ``{timestamp}{METHOD}{path}``

The private key is loaded from the path in ``KALSHI_PRIVATE_KEY_PATH`` (a PEM
file). ``path`` is the request path **including** the ``/trade-api/v2`` prefix
and any query string, e.g. ``/trade-api/v2/markets?limit=10``.
"""

from __future__ import annotations

import base64
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey


@lru_cache(maxsize=4)
def _load_private_key(pem_path: str) -> RSAPrivateKey:
    """Load (and cache) an RSA private key from a PEM file.

    Cached per path so the disk read + parse happens once per process.
    """
    data = Path(pem_path).expanduser().read_bytes()
    key = serialization.load_pem_private_key(data, password=None)
    if not isinstance(key, RSAPrivateKey):
        raise ValueError(f"Key at {pem_path} is not an RSA private key.")
    return key


def _sign(private_key: RSAPrivateKey, message: str) -> str:
    """RSA-PSS-SHA256 sign ``message`` and return a base64 string."""
    signature = private_key.sign(
        message.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("ascii")


def build_auth_headers(
    *,
    access_key_id: str,
    private_key_path: str,
    method: str,
    path: str,
    timestamp_ms: Optional[int] = None,
) -> Dict[str, str]:
    """Build the three Kalshi auth headers for a request.

    Args:
        access_key_id: Kalshi Access Key ID.
        private_key_path: Path to the RSA private-key PEM file.
        method: HTTP method, e.g. ``"GET"`` (case-insensitive; upper-cased).
        path: Request path including ``/trade-api/v2`` prefix and query string.
        timestamp_ms: Override epoch-ms timestamp (for tests); defaults to now.

    Returns:
        Dict of headers: KALSHI-ACCESS-KEY / -TIMESTAMP / -SIGNATURE.
    """
    ts = str(timestamp_ms if timestamp_ms is not None else int(time.time() * 1000))
    method_upper = method.upper()
    message = f"{ts}{method_upper}{path}"
    private_key = _load_private_key(private_key_path)
    signature = _sign(private_key, message)
    return {
        "KALSHI-ACCESS-KEY": access_key_id,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": signature,
    }


def sign_message(private_key_path: str, message: str) -> str:
    """Sign an arbitrary message string (helper for tests / advanced use)."""
    return _sign(_load_private_key(private_key_path), message)