"""Shared helpers for the test suite."""

from collections.abc import Callable
from typing import Any


def unwrap(transformer: Any) -> Callable[..., Any]:
    """
    Get the raw function behind a `dlt.transformer`.

    `dlt` rebinds arguments when a transformer is called directly, so tests
    call the underlying function instead. No dlt pipeline machinery runs.
    """
    return transformer.__wrapped__
