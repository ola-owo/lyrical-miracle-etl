"""Tests for data_loader.genius helpers and transformers."""

from datetime import date

import polars as pl
import pyarrow as pa
import pytest
import requests
from dlt.sources.helpers.requests import HTTPError

from data_loader import genius
from tests.utils import unwrap

match_to_dataset = unwrap(genius.match_to_dataset)
match_search_results = unwrap(genius.match_search_results)
get_lyrics = unwrap(genius.get_lyrics)
get_song_metadata = unwrap(genius.get_song_metadata)


@pytest.mark.parametrize(
    'raw, expected',
    [
        ('Plain Title', 'Plain Title'),
        # removing a tag leaves a trailing space (current behavior)
        ('Song (Remix)', 'Song '),
        ('Song [Bonus Track]', 'Song '),
        ('Song - Live', 'Song'),
        ('Song feat. Artist', 'Song '),
        ('Song (feat. X) [Live]', 'Song '),
        # unicode is composed to NFC
        ('Cafe\u0301', 'Café'),
    ],
)
def test_normalize_titles(raw, expected):
    assert genius.normalize_titles(pl.Series([raw])).to_list() == [expected]


def test_fuzzy_match_returns_closest_candidate():
    res = genius.fuzzy_match(
        pl.Series(['hello world']),
        pl.Series(['goodbye', 'hello world']),
    )
    assert res.shape == (1, 3)
    assert res['query'].to_list() == ['hello world']
    assert res['match_id'].to_list() == [1]
    assert res['sim'].to_list() == [100]


def test_fuzzy_match_translates_indices_to_ids():
    res = genius.fuzzy_match(
        pl.Series(['hello world']),
        pl.Series(['goodbye', 'hello world']),
        pl.Series([101, 202]),
    )
    assert res['match_id'].to_list() == [202]
    assert res['sim'].to_list() == [100]


def test_fuzzy_match_batch_mode():
    res = genius.fuzzy_match(
        pl.Series(['hello', 'goodbye']),
        pl.Series(['hello world', 'hello', 'goodbye']),
        batch=True,
    )
    assert res['query'].to_list() == ['hello', 'goodbye']
    assert res['match_id'].to_list() == [1, 2]
    assert res['sim'].to_list() == [100, 100]


def test_fuzzy_match_batch_mode_with_ids():
    res = genius.fuzzy_match(
        pl.Series(['hello']),
        pl.Series(['hello world', 'hello']),
        pl.Series([7, 8]),
        batch=True,
    )
    assert res['match_id'].to_list() == [8]


def test_fuzzy_match_rejects_empty_input():
    with pytest.raises(AssertionError):
        genius.fuzzy_match(pl.Series([], dtype=pl.String), pl.Series(['x']))
    with pytest.raises(AssertionError):
        genius.fuzzy_match(pl.Series(['x']), pl.Series([], dtype=pl.String))


def test_fuzzy_match_rejects_mismatched_ids():
    with pytest.raises(AssertionError):
        genius.fuzzy_match(pl.Series(['x']), pl.Series(['a', 'b']), pl.Series([1]))


def _write_song_dataset(path):
    pl.DataFrame(
        {
            'id': [1, 2],
            'title': ['Hello', 'Other'],
            'artist': ['World', 'Band'],
        }
    ).write_parquet(path)


def test_match_to_dataset_returns_matches(tmp_path, monkeypatch):
    dataset = tmp_path / 'genius-dset.parquet'
    _write_song_dataset(dataset)
    monkeypatch.setattr(genius, 'SONG_DATASET', dataset)

    res = match_to_dataset(pa.table({'song': ['Hello'], 'artist': ['World']}))

    assert res == [
        {
            'song': 'Hello',
            'artist': 'World',
            'g_id': 1,
            'g_title': 'Hello',
            'g_artist': 'World',
            'match_score': 100,
        }
    ]


def test_match_to_dataset_filters_below_cutoff(tmp_path, monkeypatch):
    dataset = tmp_path / 'genius-dset.parquet'
    _write_song_dataset(dataset)
    monkeypatch.setattr(genius, 'SONG_DATASET', dataset)

    res = match_to_dataset(pa.table({'song': ['Intruder'], 'artist': ['Fixture']}))

    assert res == []


def test_match_to_dataset_without_input():
    assert match_to_dataset(pa.table({'song': [], 'artist': []})) is None


def test_match_search_results_picks_best_candidate():
    search_results = pa.table(
        {
            'song': ['Hello', 'Hello'],
            'artist': ['World', 'World'],
            'searchtext': ['Hello World', 'Hello World'],
            'g_id': [1, 2],
            'g_title': ['Hello', 'Nothing'],
            'g_primary_artist_names': ['World', 'Who'],
        }
    )

    res = list(match_search_results(search_results))

    assert res == [
        [
            {
                'song': 'Hello',
                'artist': 'World',
                'g_id': 1,
                'g_title': 'Hello',
                'g_artist': 'World',
                'match_score': 100,
            }
        ]
    ]


def test_match_search_results_below_cutoff_yields_unmatched_song():
    search_results = pa.table(
        {
            'song': ['Nope'],
            'artist': ['Band'],
            'searchtext': ['Nope Band'],
            'g_id': [2],
            'g_title': ['Nothing'],
            'g_primary_artist_names': ['Who'],
        }
    )

    res = list(match_search_results(search_results))

    assert res == [[{'song': 'Nope', 'artist': 'Band'}]]


def test_match_search_results_drops_null_candidates():
    search_results = pa.table(
        {
            'song': ['Hello'],
            'artist': ['World'],
            'searchtext': ['Hello World'],
            'g_id': [None],
            'g_title': [None],
            'g_primary_artist_names': [None],
        }
    )

    assert list(match_search_results(search_results)) == []


def test_match_search_results_without_input():
    search_results = pa.table(
        {
            'song': [],
            'artist': [],
            'searchtext': [],
            'g_id': [],
            'g_title': [],
            'g_primary_artist_names': [],
        }
    )

    assert list(match_search_results(search_results)) == []


def _genius_hit(**result):
    """Build a Genius search hit. Keyword arguments override fields of `result`."""
    fields = {
        'id': 1,
        'artist_names': 'World',
        'full_title': 'Hello by World',
        'primary_artist_names': 'World',
        'title': 'Hello',
        'title_with_featured': 'Hello',
        'lyrics_state': 'complete',
        'release_date_components': {'year': 2020, 'month': 3, 'day': 4},
        'stats': {'pageviews': 10},
    }
    return {'index': 'song', 'type': 'song', 'result': {**fields, **result}}


def test_parse_genius_search_res_without_hits():
    assert genius._parse_genius_search_res([]) is None


def test_parse_genius_search_res_flattens_hit():
    res = genius._parse_genius_search_res([_genius_hit()])

    assert res.to_dicts() == [
        {
            'g_id': 1,
            'g_artist_names': 'World',
            'g_full_title': 'Hello by World',
            'g_primary_artist_names': 'World',
            'g_title': 'Hello',
            'g_title_with_featured': 'Hello',
            'g_release_date': date(2020, 3, 4),
            'g_pageviews': 10,
            'g_lyrics_complete': True,
        }
    ]


@pytest.mark.parametrize(
    'components',
    [
        None,
        {'year': 2020, 'month': None, 'day': None},
        {'year': 2020, 'month': 3, 'day': None},
        {'year': None, 'month': 3, 'day': 4},
    ],
)
def test_parse_genius_search_res_incomplete_release_date_is_null(components):
    res = genius._parse_genius_search_res(
        [_genius_hit(release_date_components=components)]
    )

    assert res['g_release_date'].to_list() == [None]
    assert res['g_pageviews'].to_list() == [10]


def test_parse_genius_search_res_release_date_per_hit():
    hits = [
        _genius_hit(id=1, release_date_components=None),
        _genius_hit(id=2),
    ]

    res = genius._parse_genius_search_res(hits)

    assert res['g_id'].to_list() == [1, 2]
    assert res['g_release_date'].to_list() == [None, date(2020, 3, 4)]


@pytest.mark.parametrize('stats', [{}, {'hot': False}, {'pageviews': None}])
def test_parse_genius_search_res_stats_without_pageviews_is_null(stats):
    res = genius._parse_genius_search_res([_genius_hit(stats=stats)])

    assert res['g_pageviews'].to_list() == [None]
    assert res['g_title'].to_list() == ['Hello']


def test_parse_genius_search_res_pageviews_per_hit():
    hits = [
        _genius_hit(id=1, stats={'hot': False}),
        _genius_hit(id=2),
    ]

    res = genius._parse_genius_search_res(hits)

    assert res['g_id'].to_list() == [1, 2]
    assert res['g_pageviews'].to_list() == [None, 10]


def _http_error(status_code):
    """Build an HTTPError with the given response status code."""
    resp = requests.Response()
    resp.status_code = status_code
    return HTTPError(f'{status_code} Client Error', response=resp)


class _FlakyGenius:
    """
    Fake lyricsgenius client.

    `responses` maps song ids to the metadata dict to return, or to an
    HTTPError to raise (wrapped in an AssertionError, like lyricsgenius
    does).
    """

    def __init__(self, responses):
        self._responses = responses

    def song(self, song_id):
        res = self._responses[song_id]
        if isinstance(res, HTTPError):
            try:
                raise res
            except HTTPError:
                raise AssertionError('request failed')
        return res


def _patch_genius_client(monkeypatch, fake):
    monkeypatch.setattr(genius, 'make_genius_client', lambda **kwargs: fake)
    monkeypatch.setattr(genius, 'sleep', lambda *args: None)


def test_get_song_metadata_trims_unneeded_fields(monkeypatch):
    song = {
        'id': 1,
        'title': 'T',
        'keep': 'yes',
        'stats': {'pageviews': 1},
        'pyongs_count': 2,
        'description_annotation': {'x': 1},
        'song_relationships': [],
        'verified_lyrics_by': [],
        'current_user_metadata': {},
    }
    _patch_genius_client(monkeypatch, _FlakyGenius({1: {'song': song}}))

    tracks = pa.table({'g_id': [1], 'g_title': ['T'], 'g_artist': ['A']})

    assert list(get_song_metadata(tracks)) == [{'id': 1, 'title': 'T', 'keep': 'yes'}]


def test_get_song_metadata_skips_failed_lookup(monkeypatch):
    _patch_genius_client(
        monkeypatch,
        _FlakyGenius({1: _http_error(404), 2: {'song': {'id': 2, 'title': 'U'}}}),
    )

    tracks = pa.table({'g_id': [1, 2], 'g_title': ['T', 'U'], 'g_artist': ['A', 'B']})

    assert list(get_song_metadata(tracks)) == [{'id': 2, 'title': 'U'}]


def test_get_song_metadata_stops_early_on_429(monkeypatch):
    _patch_genius_client(
        monkeypatch,
        _FlakyGenius({1: _http_error(429), 2: {'song': {'id': 2, 'title': 'U'}}}),
    )

    tracks = pa.table({'g_id': [1, 2], 'g_title': ['T', 'U'], 'g_artist': ['A', 'B']})

    assert list(get_song_metadata(tracks)) == []


def test_get_song_metadata_without_input(monkeypatch):
    _patch_genius_client(monkeypatch, _FlakyGenius({}))

    tracks = pa.table({'g_id': [], 'g_title': [], 'g_artist': []})

    assert list(get_song_metadata(tracks)) == []


class _FakeLyricsGenius:
    def __init__(self, lyrics):
        self._lyrics = lyrics

    def lyrics(self, song_url=None, song_id=None, remove_section_headers=False):
        return self._lyrics


def _songs_table():
    return pa.table(
        {
            'id': [1, 2, 3],
            'full_title': ['A', 'B', 'C'],
            'lyrics_state': ['complete', 'complete', 'unreleased'],
        }
    )


def test_get_lyrics_returns_lyrics(monkeypatch):
    _patch_genius_client(monkeypatch, _FakeLyricsGenius('la la'))

    assert list(get_lyrics(_songs_table())) == [
        {'id': 1, 'lyrics': 'la la'},
        {'id': 2, 'lyrics': 'la la'},
        {'id': 3, 'lyrics': 'la la'},
    ]


def test_get_lyrics_skips_incomplete_songs_without_lyrics(monkeypatch):
    _patch_genius_client(monkeypatch, _FakeLyricsGenius(None))

    res = list(get_lyrics(_songs_table()))

    # songs 1 and 2 have complete lyrics, so they are likely instrumentals;
    # song 3 is 'unreleased' and is left to be searched again
    assert res == [{'id': 1, 'lyrics': None}, {'id': 2, 'lyrics': None}]


class _FlakyLyricsGenius:
    """Fake lyricsgenius client raising HTTPErrors per song id."""

    def __init__(self, responses):
        self._responses = responses

    def lyrics(self, song_url=None, song_id=None, remove_section_headers=False):
        res = self._responses[song_id]
        if isinstance(res, HTTPError):
            try:
                raise res
            except HTTPError:
                raise AssertionError('request failed')
        return res


def test_get_lyrics_stops_early_on_429(monkeypatch):
    fake = _FlakyLyricsGenius({1: _http_error(429), 2: 'la la'})
    _patch_genius_client(monkeypatch, fake)

    songs = pa.table(
        {
            'id': [1, 2],
            'full_title': ['A', 'B'],
            'lyrics_state': ['complete', 'complete'],
        }
    )

    assert list(get_lyrics(songs)) == []


def test_get_lyrics_without_input(monkeypatch):
    _patch_genius_client(monkeypatch, _FakeLyricsGenius('la la'))

    songs = pa.table({'id': [], 'full_title': [], 'lyrics_state': []})

    assert list(get_lyrics(songs)) == []
