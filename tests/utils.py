"""Shared helpers for the test suite."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import dagster as dg


def unwrap(transformer: Any) -> Callable[..., Any]:
    """
    Get the raw function behind a `dlt.transformer`.

    `dlt` rebinds arguments when a transformer is called directly, so tests
    call the underlying function instead. No dlt pipeline machinery runs.
    """
    return transformer.__wrapped__


def unwrap_check(
    checks_def: dg.AssetChecksDefinition,
) -> Callable[..., dg.AssetCheckResult]:
    """Get the raw function behind a `@dg.asset_check` definition."""
    return checks_def.node_def.compute_fn.decorated_fn


class FakeResource:
    """Chainable stand-in for a dlt resource or source."""

    def __init__(self, name: str = 'fake-resource'):
        self.name = name
        self.call: SimpleNamespace | None = None
        self.steps: list[tuple[str, Any]] = []
        self.appended: list[Any] = []

    def add_filter(self, fn: Callable[[Any], bool]) -> FakeResource:
        self.steps.append(('filter', fn))
        return self

    def add_map(self, fn: Callable[[Any], Any]) -> FakeResource:
        self.steps.append(('map', fn))
        return self

    def add_limit(self, limit: int) -> FakeResource:
        self.steps.append(('limit', limit))
        return self

    def __or__(self, other: Any) -> FakeResource:
        self.appended.append(other)
        return self

    def filters(self) -> list[Callable[[Any], bool]]:
        return [fn for kind, fn in self.steps if kind == 'filter']

    def limits(self) -> list[int]:
        return [limit for kind, limit in self.steps if kind == 'limit']

    def step_names(self) -> list[str]:
        return [kind for kind, _ in self.steps]


class FakePipeline:
    """Stand-in for a dlt pipeline that records calls instead of loading."""

    def __init__(self, row_counts: dict[str, int] | None = None):
        self.runs: list[SimpleNamespace] = []
        self.extracts: list[SimpleNamespace] = []
        self.normalize_calls = 0
        if row_counts is None:
            self.last_trace = None
        else:
            self.last_trace = SimpleNamespace(
                last_normalize_info=SimpleNamespace(row_counts=row_counts)
            )

    def run(self, source: Any, **kwargs: Any) -> FakePipeline:
        self.runs.append(SimpleNamespace(source=source, kwargs=kwargs))
        return self

    def extract(self, source: Any, **kwargs: Any) -> FakePipeline:
        self.extracts.append(SimpleNamespace(source=source, kwargs=kwargs))
        return self

    def normalize(self) -> FakePipeline:
        self.normalize_calls += 1
        return self


class PipelineRecorder(list):
    """
    Pipelines created while a test runs.

    Set `row_counts` before materializing to fake dlt's normalize row counts.
    """

    row_counts: dict[str, int] | None = None
