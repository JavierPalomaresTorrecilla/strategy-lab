"""Narrow unauthenticated wrapper around Coinbase's public REST endpoints."""

from typing import Any

_RESTClient: type[Any] | None = None


class CoinbasePublicClient:
    """Expose only Coinbase REST endpoints that are safe for public data access."""

    __slots__ = ("_client",)

    def __init__(self, client: Any) -> None:
        self._client = client

    def get_unix_time(self) -> Any:
        return self._client.get_unix_time()

    def get_public_product(self, product_id: str) -> Any:
        return self._client.get_public_product(product_id)

    def get_public_products(self, **kwargs: Any) -> Any:
        return self._client.get_public_products(**kwargs)

    def get_public_candles(
        self,
        product_id: str,
        start: str,
        end: str,
        granularity: str,
        limit: int | None = None,
    ) -> Any:
        return self._client.get_public_candles(
            product_id, start, end, granularity, limit=limit
        )


def make_public_client(
    timeout: float = 10.0, *, rate_limit_headers: bool = False
) -> CoinbasePublicClient:
    """Construct an explicitly unauthenticated Coinbase REST client."""
    global _RESTClient

    if not isinstance(rate_limit_headers, bool):
        raise TypeError("rate_limit_headers must be a bool")

    if _RESTClient is None:
        try:
            from coinbase.rest import RESTClient
        except ImportError as exc:
            raise ImportError(
                "Coinbase support requires the optional `crypto` extra; "
                "install it with `pip install .[crypto]`."
            ) from exc
        _RESTClient = RESTClient

    client = _RESTClient(
        api_key=None,
        api_secret=None,
        key_file=None,
        timeout=timeout,
        rate_limit_headers=rate_limit_headers,
    )
    if client.is_authenticated is not False:
        raise RuntimeError("Coinbase public client construction unexpectedly authenticated.")

    return CoinbasePublicClient(client)
