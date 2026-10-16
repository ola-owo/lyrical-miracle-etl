"""Tests for data_loader.checks asset checks."""

from unittest.mock import MagicMock

import pyarrow as pa
import pytest

from data_loader import big5, checks
from tests.utils import unwrap_check


def _run(checks_def):
    return unwrap_check(checks_def)()


def _patch_query_count(monkeypatch, count, queries=None):
    def fake_query_count(query):
        if queries is not None:
            queries.append(query)
        return count

    monkeypatch.setattr(checks, '_query_count', fake_query_count)


def test_query_count_returns_first_value(monkeypatch):
    client = MagicMock()
    client.query_and_wait.return_value.to_arrow.return_value = pa.table({'n': [3]})
    client_cls = MagicMock(return_value=client)
    monkeypatch.setattr(checks.bigquery, 'Client', client_cls)

    assert checks._query_count('SELECT 42') == 3

    client_cls.assert_called_once_with(project='test-project')
    client.query_and_wait.assert_called_once_with('SELECT 42')


EMBEDDING_CHECKS = [
    pytest.param(checks.all_checks[0], 'lyrics_embed', id='plain'),
    pytest.param(checks.all_checks[1], 'lyrics_embed_clustering', id='clustering'),
]


@pytest.mark.parametrize('checks_def, table', EMBEDDING_CHECKS)
def test_embedding_dim_check_passes_and_queries_table(monkeypatch, checks_def, table):
    queries = []
    _patch_query_count(monkeypatch, 0, queries)

    result = _run(checks_def)

    assert result.passed is True
    (query,) = queries
    assert f'FROM genius.{table}' in query
    assert f'ARRAY_LENGTH(embedding) != {checks.EMBEDDING_DIM}' in query


@pytest.mark.parametrize('checks_def, table', EMBEDDING_CHECKS)
def test_embedding_dim_check_fails_on_bad_rows(monkeypatch, checks_def, table):
    _patch_query_count(monkeypatch, 2)

    result = _run(checks_def)

    assert result.passed is False
    assert result.metadata['rows_with_wrong_dim'].value == 2


def test_cluster_labels_check_passes_when_all_labels_have_centroids(monkeypatch):
    queries = []
    _patch_query_count(monkeypatch, 0, queries)

    result = _run(checks.cluster_labels_have_centroids)

    assert result.passed is True
    (query,) = queries
    assert 'FROM genius.lyrics_clusters' in query
    assert 'LEFT JOIN genius.lyrics_cluster_centroids' in query


def test_cluster_labels_check_fails_on_orphan_labels(monkeypatch):
    _patch_query_count(monkeypatch, 3)

    result = _run(checks.cluster_labels_have_centroids)

    assert result.passed is False
    assert result.metadata['unmatched_labels'].value == 3


def test_big5_traits_check_requires_all_traits(monkeypatch):
    queries = []
    _patch_query_count(monkeypatch, 0, queries)

    result = _run(checks.big5_traits_are_complete)

    assert result.passed is True
    (query,) = queries
    assert 'FROM genius.lyrics_big5' in query
    for trait in big5.BIG5_TRAITS_SHORT:
        assert f'{trait} IS NULL' in query


def test_big5_traits_check_fails_on_incomplete_rows(monkeypatch):
    _patch_query_count(monkeypatch, 1)

    result = _run(checks.big5_traits_are_complete)

    assert result.passed is False
    assert result.metadata['incomplete_rows'].value == 1
