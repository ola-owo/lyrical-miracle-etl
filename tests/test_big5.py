"""Tests for data_loader.big5 helpers and transformers."""

from types import SimpleNamespace

import polars as pl
import pytest

from data_loader import big5
from tests.utils import unwrap

big5_predict_online = unwrap(big5.big5_predict_online)


def test_check_predict_inputs_accepts_clean_batch():
    texts = pl.DataFrame({'id': [1], 'content': ['hello']})
    assert big5._check_predict_inputs(texts) is None


def test_check_predict_inputs_rejects_nulls():
    texts = pl.DataFrame({'id': [1, 2], 'content': ['hello', None]})
    with pytest.raises(ValueError, match='Null batch inputs'):
        big5._check_predict_inputs(texts)


class FakeEndpoint:
    """Records predict calls and returns constant trait scores."""

    def __init__(self):
        self.calls = []

    def predict(self, instances, parameters=None):
        self.calls.append(instances)
        return SimpleNamespace(predictions=[[0.5] * 5 for _ in instances])


def test_big5_ping_sends_empty_request():
    endpoint = FakeEndpoint()
    big5.big5_ping(endpoint=endpoint)
    assert endpoint.calls == [[]]


def test_big5_predict_online_predicts_traits():
    endpoint = FakeEndpoint()
    texts = pl.DataFrame({'id': [1, 2], 'content': ['a', 'b']})

    res = list(big5_predict_online(texts, endpoint=endpoint))

    assert res == [
        [
            {'id': 1, 'OPN': 0.5, 'CON': 0.5, 'EXT': 0.5, 'AGR': 0.5, 'NEU': 0.5},
            {'id': 2, 'OPN': 0.5, 'CON': 0.5, 'EXT': 0.5, 'AGR': 0.5, 'NEU': 0.5},
        ]
    ]
    assert endpoint.calls == [['a', 'b']]


def test_big5_predict_online_splits_requests():
    endpoint = FakeEndpoint()
    texts = pl.DataFrame(
        {'id': list(range(9)), 'content': [f'text {i}' for i in range(9)]}
    )

    list(big5_predict_online(texts, endpoint=endpoint))

    assert [len(call) for call in endpoint.calls] == [8, 1]


def test_big5_predict_online_without_input():
    endpoint = FakeEndpoint()
    texts = pl.DataFrame({'id': [], 'content': []})

    assert list(big5_predict_online(texts, endpoint=endpoint)) == []
    assert endpoint.calls == []
