"""Shared fixtures for the test suite."""

from types import SimpleNamespace

import dlt
import pytest

from data_loader import big5, embeddings, genius
from tests.utils import FakePipeline, FakeResource, PipelineRecorder

# only the dlt secrets/config touched by the code under test; tests should
# not depend on a developer's local .dlt directory
DUMMY_DLT_CONFIG = {
    'SOURCES__BIGQUERY__CREDENTIALS': 'test-bigquery-credentials',
    'SOURCES__GEMINI__API_KEY': 'test-gemini-key',
    'GCLOUD__PROJECT_ID': 'test-project',
    'GCLOUD__REGION': 'test-region',
    'GCLOUD__BIG5_ENDPOINT': 'test-big5-endpoint',
}


@pytest.fixture(autouse=True)
def dummy_dlt_config(monkeypatch):
    """Give dlt dummy config/secrets so tests pass without a local .dlt dir."""
    for name, value in DUMMY_DLT_CONFIG.items():
        monkeypatch.setenv(name, value)


@pytest.fixture
def fake_pipelines(monkeypatch) -> PipelineRecorder:
    """Patch `dlt.pipeline` to record created pipelines instead of running them."""
    recorder = PipelineRecorder()

    def make_pipeline(*args, **kwargs):
        pipeline = FakePipeline(row_counts=recorder.row_counts)
        recorder.append(SimpleNamespace(args=args, kwargs=kwargs, pipeline=pipeline))
        return pipeline

    monkeypatch.setattr(dlt, 'pipeline', make_pipeline)
    return recorder


@pytest.fixture
def fake_sql_table(monkeypatch) -> list[FakeResource]:
    """Patch `sql_table` in every module that imports it."""
    created: list[FakeResource] = []

    def make_table(*args, **kwargs):
        resource = FakeResource(name=str(kwargs.get('table', 'sql_table')))
        resource.call = SimpleNamespace(args=args, kwargs=kwargs)
        created.append(resource)
        return resource

    for module in (genius, big5, embeddings):
        monkeypatch.setattr(module, 'sql_table', make_table)
    return created
