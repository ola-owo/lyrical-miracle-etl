"""Tests for data_loader.lastfm helpers."""

import pendulum as pn
import pytest

from data_loader.lastfm import convert_uts, filter_now_playing, fix_text_fields


def test_convert_uts_replaces_date_with_datetime():
    track = {'date': {'uts': '1700000000', '#text': '14 Nov 2023'}, 'name': 'song'}

    res = convert_uts(track)

    assert res is track
    assert res['dt'] == pn.from_timestamp(1700000000)
    assert res['name'] == 'song'
    assert 'date' not in res


def test_convert_uts_accepts_int_uts():
    track = {'date': {'uts': 1700000000}}
    assert convert_uts(track)['dt'] == pn.from_timestamp(1700000000)


def test_fix_text_fields_flattens_artist_and_album():
    track = {
        'artist': {'#text': 'Radiohead', 'mbid': 'artist-mbid'},
        'album': {'#text': 'In Rainbows', 'mbid': 'album-mbid'},
        'name': 'Weird Fishes',
        'streamable': '0',
    }

    res = fix_text_fields(track)

    assert res['artist'] == 'Radiohead'
    assert res['artist__mbid'] == 'artist-mbid'
    assert res['album'] == 'In Rainbows'
    assert res['album__mbid'] == 'album-mbid'
    assert res['song'] == 'Weird Fishes'
    assert 'name' not in res
    assert 'streamable' not in res


def test_fix_text_fields_missing_mbid_becomes_none():
    track = {
        'artist': {'#text': 'Radiohead'},
        'album': {'#text': 'In Rainbows'},
        'name': 'Weird Fishes',
        'streamable': '0',
    }

    res = fix_text_fields(track)

    assert res['artist__mbid'] is None
    assert res['album__mbid'] is None


@pytest.mark.parametrize(
    'track, expected',
    [
        ({}, True),
        ({'@attr': {}}, True),
        ({'@attr': {'nowplaying': 'true'}}, False),
        ({'@attr': {'nowplaying': False}}, True),
        ({'name': 'song', '@attr': {'nowplaying': 'true'}}, False),
    ],
)
def test_filter_now_playing(track, expected):
    assert filter_now_playing(track) is expected
