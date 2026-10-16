"""Tests for data_loader.dlt_utils."""

from types import SimpleNamespace

from data_loader.dlt_utils import get_normalize_row_counts


def _pipeline(trace):
    return SimpleNamespace(last_trace=trace)


def _trace(normalize_info):
    return SimpleNamespace(last_normalize_info=normalize_info)


def test_returns_empty_dict_without_trace():
    assert get_normalize_row_counts(_pipeline(None)) == {}


def test_returns_empty_dict_without_normalize_info():
    assert get_normalize_row_counts(_pipeline(_trace(None))) == {}


def test_returns_empty_dict_without_row_counts():
    normalize_info = SimpleNamespace(row_counts=None)
    assert get_normalize_row_counts(_pipeline(_trace(normalize_info))) == {}


def test_returns_row_counts():
    normalize_info = SimpleNamespace(row_counts={'scrobbles': 42, 'songs': 7})
    res = get_normalize_row_counts(_pipeline(_trace(normalize_info)))
    assert res == {'scrobbles': 42, 'songs': 7}
