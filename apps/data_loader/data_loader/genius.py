from pathlib import Path
import logging
from time import sleep
from random import random

import polars as pl
import polars.selectors as cs
import pyarrow as pa
from tqdm import tqdm
import numpy as np
from scipy import sparse
import rapidfuzz as fuzz
from lyricsgenius import Genius

from airflow.sdk import task
import dlt
from dlt import transformer
from dlt.sources.sql_database import sql_table
from dlt.sources.helpers.requests.retry import Client
from dlt.sources.helpers.requests import HTTPError


_genius_client_excluded_terms = [
    '(live)',
    '(remix)',
    'instrumental',
    # exclude translations:
    'tradução',
    'traduções',
    'traduccion',
    'traducción',
    'traducciones',
    'traducciónes',
    'traduction',
    'traductions',
    'traduzione',
    'traduzioni',
    'vertaling',
    'vertalingen',
    'Übersetzung',
    'Ubersetzung',
    'Übersetzungen',
    'Ubersetzungen',
    'перевод',
    'переводы',
    'переклад',
    'переклади',
    'çeviri',
    'çeviriler',
    'अनुवाद',
    '翻译',
    '翻訳',
]


def make_genius_client(
    remove_section_headers=True,
    skip_non_songs=False,
    excluded_terms=_genius_client_excluded_terms,
    sleep_time=1,
    timeout=15,
    retries=1,
    **kwargs,
) -> Genius:
    """Build the lyricsgenius client"""
    excluded_terms = excluded_terms.copy()
    excluded_terms.remove('instrumental')
    client = Genius(
        dlt.secrets['sources.genius.token'],
        remove_section_headers=remove_section_headers,
        skip_non_songs=skip_non_songs,
        excluded_terms=excluded_terms,
        sleep_time=sleep_time,
        timeout=timeout,
        retries=retries,
        **kwargs,
    )
    # EXPERIMENTAL: replace lyricsgenius requests session with dlt's session
    # https://dlthub.com/docs/api_reference/dlt/sources/helpers/requests/session
    client._session = Client(
        request_timeout=10,
        respect_retry_after_header=True,
        session_attrs=dict(headers=client._session.headers),
    )
    return client


def normalize_titles(strings):
    """
    normalize a polars string column/series,
    meant for song titles with extra tags like "feat. (...)" or "(remix)"
    """
    return (
        strings.str.normalize()
        .str.replace(r'\(.*\)', '')
        .str.replace(r'\[.*\]', '')
        .str.replace(r'(?i)feat\.? .+$', '')
        .str.replace(r' - .*$', '')
        # .str.replace_all(r'[^\w\s]', '') # remove punctuation
        .str.replace_all(r'\s+', ' ')
    )


def fuzzy_match(
    queries,
    candidates,
    candidate_ids=None,
    batch=False,
    cutoff=80,
    scorer=fuzz.fuzz.WRatio,
):
    """
    Fuzzy match stringing.
    Test all combinations of `queries` and `candidates` and return the closest match for each query.

    :param queries: strings to query
    :type queries: polars.Series
    :param candidates: candidate matches
    :type candidates: polars.Series
    :param candidate_ids: ids of candidate matches.
        if specified, return these IDs instead of indices of matches
    :type candidate_ids: polars.Series
    :param batch: if false (default), search one query at a time.
        otherwise search all queries at once
    :type batch: bool
    :param cutoff: when `batch=True`, minimum similarity (out of 100) required to keep a match.
    :type cutoff: int
    :param scorer: fuzzy scoring algorithm to use

    :returns: Dataframe with columns (`query`, `match_id`, and `sim`)
    :rtype: polars.DataFrame
    """
    assert len(queries) > 0
    assert len(candidates) > 0
    candidates_have_ids = candidate_ids is not None
    assert not (candidates_have_ids and len(candidate_ids) != len(candidates))

    if batch:
        search_res = sparse.coo_array(
            fuzz.process.cdist(
                queries,
                candidates,
                processor=fuzz.utils.default_process,
                dtype=np.uint8,
                workers=-1,
                scorer=scorer,
                score_cutoff=cutoff,
            )
        )
        search_res_df = pl.DataFrame(
            {
                'query': queries,
                'match_id': search_res.argmax(axis=1),
                'sim': search_res.max(axis=1).toarray(),
            }
        )
        if candidates_have_ids:
            # translate matched song indices to ids
            match_ids = candidate_ids[search_res_df['match_id']]
            search_res_df = search_res_df.with_columns(match_id=match_ids)
        return search_res_df
    else:
        search_res = (
            fuzz.process.extractOne(
                query,
                candidates,
                processor=fuzz.utils.default_process,
                scorer=scorer,
            )
            for query in queries
        )
        schema = [('match', pl.String), ('sim', pl.UInt8), ('match_id', pl.UInt64)]
        search_res_df = (
            pl.from_records(list(search_res), schema=schema)
            .with_columns(query=pl.Series(queries))
            .select('query', 'match_id', 'sim')
        )
        if candidates_have_ids:
            # translate matched song indices to ids
            match_ids = candidate_ids[search_res_df['match_id']]
            search_res_df = search_res_df.with_columns(match_id=match_ids)
        return search_res_df


@transformer
def match_to_dataset(search_queries: pa.Table):
    """
    Match scrobbles to the genius dataset

    :param search_queries: table with `song` and `artist` columns
    """
    FUZZY_MATCH_SIM_CUTOFF = 96  # from visual inspection of a recent sample
    SONG_DATASET = Path('data/genius-no-lyrics.parquet')

    log = logging.getLogger('dlt')
    if search_queries.num_rows == 0:
        log.info('no songs given, exiting.')
        return

    # not normalizing Genius titles bc they're usually already clean,
    # and so we can tell instrumentals apart from real songs
    genius_titles = (
        pl.scan_parquet(SONG_DATASET)
        .with_columns(searchtext=pl.col('title') + ' ' + pl.col('artist'))
        .select('id', 'title', 'artist', 'searchtext')
    )

    search_queries = pl.from_arrow(search_queries)
    search_queries = search_queries.select(
        'song',
        'artist',
        searchtext=normalize_titles(pl.col('song')) + ' ' + pl.col('artist'),
    )

    log.info(f'Checking {search_queries.height} songs against the genius dataset')
    search_results = fuzzy_match(
        search_queries['searchtext'].unique(),
        genius_titles.select('searchtext').collect().to_series(),
        genius_titles.select('id').collect().to_series(),
        batch=True,
    )

    # save fuzzy-match results (for debugging)
    # import pickle, gzip
    # with gzip.open('search_results.pkl.gz', 'wb') as f:
    #     pickle.dump(search_results, f)
    # with gzip.open('search_results.pkl.gz', 'rb') as f:
    #     search_results = pickle.load(f)

    matches = (
        search_results.lazy()
        .filter(pl.col('sim') >= FUZZY_MATCH_SIM_CUTOFF)
        # recover original song/artist names and matched song ids
        .join(
            search_queries.lazy(),
            left_on='query',
            right_on='searchtext',
            how='inner',
            validate='1:m',
        )
        .join(
            genius_titles,
            left_on='match_id',
            right_on='id',
            how='inner',
            suffix='_g',
            validate='m:1',
        )
        .rename(
            {
                'artist_g': 'g_artist',
                'sim': 'match_score',
                'match_id': 'g_id',
                'title': 'g_title',
            }
        )
        .select('song', 'artist', cs.starts_with('g_'), 'match_score')
    )
    return matches.collect().to_arrow()


@task
def match_to_dataset_task():
    """
    Find newly added scrobbles and try to match them against the Genius dataset.
    Write the results to `lastfm.genius_matches`

    Returns: number of loaded records
    """
    log = logging.getLogger('airflow.task')
    unsearched_songs = sql_table(
        dlt.secrets['sources.postgres.credentials'],
        table='songs_not_searched',
        schema='lastfm',
        backend='connectorx',
        backend_kwargs={'conn': dlt.secrets['sources.postgres.credentials']},
    )

    pipeline = dlt.pipeline('genius_search', destination=dlt.destinations.postgres())
    pipeline.run(
        unsearched_songs | match_to_dataset,
        dataset_name='lastfm',
        table_name='genius_matches',
        write_disposition='merge',
        primary_key=('song', 'artist'),
        loader_file_format='parquet',
    )

    row_counts = pipeline.last_trace.last_normalize_info.row_counts or {}
    log.info(f'row counts: {row_counts}')
    return row_counts.get('genius_matches', 0)


@transformer
def genius_search(tracks: pa.Table):
    """
    Search Genius for the given song

    Input: Table containing songs to search. Must have columns `song` and `artist`

    Yields: Search result(s), no result will have null `g_id`
    """

    def parse_genius_search_res(hits) -> pl.DataFrame|None:
        """
        Extract song info from genius search results.
        Return a dict, or None if there are no results

        Remember to add other genius_search_results fields (song,artist,searchtext)
        after calling this function
        """
        if not hits:
            return None

        return (
            pl.from_dicts(hits)
            .select(pl.col('result').struct.unnest())
            .with_columns(
                pl.col('release_date_components').replace(None, pl.struct(None)),
                pl.col('stats').replace(None, pl.struct(None)),
            )
            .select(
                'id',
                'artist_names',
                'full_title',
                'primary_artist_names',
                'title',
                'title_with_featured',
                pl.date(
                    pl.col('release_date_components').struct.field('^year$'),
                    pl.col('release_date_components').struct.field('^month$'),
                    pl.col('release_date_components').struct.field('^day$'),
                ).alias('release_date'),
                pl.col('stats').struct.field('^pageviews$'),
                pl.col('lyrics_state').eq('complete').alias('lyrics_complete'),
            )
            .select(cs.all().name.prefix('g_'))
        )

    log = logging.getLogger('dlt')
    if tracks.num_rows == 0:
        log.info('no search queries given, exiting.')
        return

    tracks: pl.DataFrame = (
        pl.from_arrow(tracks)
        .with_columns(
            searchtext=normalize_titles(pl.col('song')) + ' ' + pl.col('artist')
        )
    )

    genius = make_genius_client()
    for track in tqdm(
        tracks.iter_rows(named=True),
        total=tracks.height,
        desc='searching for tracks',
        leave=False,
        unit='req',
    ):
        no_result = {
            # blank record indicates no search results
            'song': track['song'],
            'artist': track['artist'],
            'searchtext': track['searchtext'],
        }
        res = genius.search_songs(track['searchtext'], per_page=5)
        search_results = parse_genius_search_res(res['hits'])
        if search_results is None:
            log.warning(f'No search results for {track["song"]} by {track["artist"]}')
            yield no_result
            continue
        yield search_results.with_columns(
            pl.lit(track['song']).alias('song'),
            pl.lit(track['artist']).alias('artist'),
            pl.lit(track['searchtext']).alias('searchtext'),
        ).to_dicts()


@task
def genius_search_task():
    """
    Search Genius for any newly scrobbled songs.
    Songs that have already been searched in Genius are ignored.
    Write search results to `lastfm.genius_searches`

    Returns: number of loaded records
    """
    log = logging.getLogger('airflow.task')
    pipeline = dlt.pipeline('genius_search', destination=dlt.destinations.postgres())
    unsearched_songs = sql_table(
        dlt.secrets['sources.postgres.credentials'],
        table='songs_not_searched',
        schema='lastfm',
        backend='connectorx',
        backend_kwargs={'conn': dlt.secrets['sources.postgres.credentials']},
    )
    pipeline.run(
        unsearched_songs | genius_search,
        dataset_name='lastfm',
        table_name='genius_searches',
        write_disposition='append',
        loader_file_format='parquet',
    )
    row_counts = pipeline.last_trace.last_normalize_info.row_counts or {}
    log.info(f'row counts: {row_counts}')
    return row_counts.get('genius_searches', 0)


@transformer
def match_search_results(search_results: pa.Table):
    """
    Match scrobbles to their genius search results.
    If no match, the yielded record will have null `g_id`.

    :param search_results: table containing all results from a single search
        (uses columns `song`, `artist`, `searchtext`, `g_id`, `g_title`, `g_artist`)
    """
    FUZZY_MATCH_SIM_CUTOFF = 90  # decent cutoff from visual inspection

    log = logging.getLogger('dlt')
    if search_results.num_rows == 0:
        log.info('no search results given, exiting.')
        return

    # not normalizing Genius titles bc they're usually already clean,
    # and so we can tell instrumentals apart from real songs
    search_results: pl.DataFrame = (
        pl.from_arrow(search_results)
        .drop_nulls(['g_id', 'searchtext'])
        .with_columns(
            g_searchtext=pl.col('g_title') + ' ' + pl.col('g_primary_artist_names')
        )
    )

    for search_results_this_song in tqdm(
        search_results.partition_by('searchtext'),
        desc='matching songs to search results',
        leave=False,
    ):
        match_results = fuzzy_match(
            search_results_this_song['searchtext'].head(1),
            search_results_this_song['g_searchtext'],
            search_results_this_song['g_id'],
            batch=False,
        )
        if match_results[0, 'sim'] < FUZZY_MATCH_SIM_CUTOFF:
            no_match = search_results_this_song.select('song', 'artist').head(1)
            yield no_match.to_dicts()
            continue
        matches = (
            search_results_this_song.join(
                match_results, how='inner', left_on='g_id', right_on='match_id'
            )
            .head(1)  # in case there are duplicate search results
            .select(
                pl.col('song'),
                pl.col('artist'),
                pl.col('g_id'),
                pl.col('g_title'),
                pl.col('g_primary_artist_names').alias('g_artist'),
                pl.col('sim').alias('match_score'),
            )
        )
        yield matches.to_dicts()


@task
def match_search_results_task():
    """
    Match scrobbles to their Genius search results.
    Write results to `lastfm.genius_matches`

    Returns: number of loaded records
    """
    log = logging.getLogger('airflow.task')
    pipeline = dlt.pipeline(
        'genius_search', destination=dlt.destinations.postgres(), dataset_name='lastfm'
    )
    unmatched_searches = sql_table(
        dlt.secrets['sources.postgres.credentials'],
        table='searches_not_matched',
        schema='lastfm',
        backend='connectorx',
        backend_kwargs={'conn': dlt.secrets['sources.postgres.credentials']},
    )
    pipeline.run(
        unmatched_searches | match_search_results,
        table_name='genius_matches',
        write_disposition='merge',
        primary_key=('song', 'artist'),
        loader_file_format='parquet',
    )
    row_counts = pipeline.last_trace.last_normalize_info.row_counts or {}
    log.info(f'row counts: {row_counts}')
    return row_counts.get('genius_matches', 0)


@transformer
def get_song_metadata(songs):
    """
    Get full song data from Genius API

    Input table columns:
        g_id: Genius song id
        g_title: song title (without artist)
        g_artist: primary artist name(s)
    """

    def _trim_song_metadata(song):
        """
        Remove unneeded fields from Genius song metadata.
        This is to save space and stop dlt from generating too many nested tables.
        """
        FIELDS_TO_RM = [
            # 'custom_performances',  # publishers, distributors, etc
            'song_relationships',  # samples, interpolations, covers, etc
            'description_annotation',
            'lyrics_marked_complete_by',
            'lyrics_marked_staff_approved_by',
            'verified_annotations_by',
            'verified_contributors',
            'verified_lyrics_by',
            'current_user_metadata',
            'stats',
            'pyongs_count',
        ]
        for f in FIELDS_TO_RM:
            song.pop(f, None)
        return song

    log = logging.getLogger('dlt')
    if songs.num_rows == 0:
        log.info('No songs to lookup, exiting.')
        return

    df_genius_song_matches = pl.from_arrow(songs)
    n_search = df_genius_song_matches.height
    log.info(f'Looking up {n_search} songs')

    gen = make_genius_client(sleep_time=0.5)
    for song in tqdm(
        df_genius_song_matches.iter_rows(named=True),
        total=n_search,
        desc='getting Genius song info',
        leave=False,
        unit='req',
    ):
        song_id = song['g_id']
        song_title = song['g_title']
        song_artist = song['g_artist']

        try:
            sleep(random())  # add 0-1s to client sleep_time
            res = gen.song(song_id)  # API CALL (public API, 10k/day limit)
        except AssertionError as e:
            if isinstance(e.__context__, HTTPError):
                e = e.__context__
                if e.response.status_code == 429:
                    log.error('Got 429 (too many requests), stopping early')
                    return
                else:  # usually 404
                    log.error(
                        '%s (%s) request failed: %s',
                        f'/songs/{song_id}',
                        f'{song_title} by {song_artist}',
                        str(e),
                    )
                    continue
            else:
                raise e

        if not res:
            log.error(
                '%s (%s) returned empty',
                f'/songs/{song_id}',
                f'{song_title} by {song_artist}',
            )
            continue
        song = res['song']
        log.debug(f'Song fields pre-trim: {list(song.keys())}')
        song = _trim_song_metadata(res['song'])
        log.debug(f'Song fields post-trim: {list(song.keys())}')
        yield song


@task
def get_song_metadata_task():
    """
    Get full song metadata for songs that don't already have it.
    Write results to `genius.songs`

    Returns: number of loaded records
    """
    log = logging.getLogger('airflow.task')
    pipeline = dlt.pipeline(
        'genius_song_meta',
        destination=dlt.destinations.postgres(),
        dataset_name='genius',
        import_schema_path='schemas/import',
        export_schema_path='schemas/export',
    )
    songs_to_search = sql_table(
        dlt.secrets['sources.postgres.credentials'],
        table='genius_matches_without_metadata',
        schema='lastfm',
        backend='connectorx',
        backend_kwargs={'conn': dlt.secrets['sources.postgres.credentials']},
    )
    pipeline.run(
        songs_to_search | get_song_metadata,
        table_name='songs',
        write_disposition='merge',
        primary_key='id',
        loader_file_format='parquet',
    )
    row_counts = pipeline.last_trace.last_normalize_info.row_counts or {}
    log.info(f'row counts: {row_counts}')
    return row_counts.get('songs', 0)


@task
def recheck_incomplete_songs_task():
    """
    Re-retrieve song metadata for songs not tagged as having complete lyrics.
    Write results to `genius.songs`

    Returns: number of loaded records
    """
    log = logging.getLogger('airflow.task')
    pipeline = dlt.pipeline(
        'genius_incomplete_song_meta',
        destination=dlt.destinations.postgres(),
        dataset_name='genius',
    )
    songs_to_search = sql_table(
        dlt.secrets['sources.postgres.credentials'],
        table='songs_incomplete',
        schema='genius',
        included_columns=['id', 'title', 'primary_artist_names'],
        backend='connectorx',
        backend_kwargs={'conn': dlt.secrets['sources.postgres.credentials']},
    )
    pipeline.run(
        songs_to_search.add_map(
            lambda t: t.rename_columns(['g_id', 'g_title', 'g_artist'])
        ) | get_song_metadata,
        table_name='songs',
        write_disposition='merge',
        primary_key='id',
    )
    row_counts = pipeline.last_trace.last_normalize_info.row_counts or {}
    log.info(f'row counts: {row_counts}')
    return row_counts.get('songs', 0)


@transformer
def get_lyrics(songs):
    """
    Get song lyrics

    Input table columns used:
    - `id`: genius id
    - `full_title`: full title with artist
    - `lyrics_state`: `"complete"` or `"unreleased"`
    """
    log = logging.getLogger('dlt')
    if songs.num_rows == 0:
        log.info('no songs to lookup, exiting.')
        return
    songs = pl.from_arrow(songs)

    # TODO (maybe): filter out old songs
    # release_date_cutoff = pn.Date.today().subtract(days=180)
    # df_lyrics = df_lyrics.filter((pl.col('g_release_date') > release_date_cutoff)
    #                              | pl.col('g_release_date').is_null())

    genius = make_genius_client(sleep_time=2)
    for song in tqdm(
        songs.iter_rows(named=True),
        total=songs.height,
        desc='getting lyrics',
        leave=False,
        unit='req',
    ):
        song_id = song['id']
        song_full_title = song['full_title']
        lyrics_complete = song['lyrics_state']
        try:
            lyrics = genius.lyrics(
                song_url=song.get('path'), song_id=song_id, remove_section_headers=True
            )
        except AssertionError as e:
            if isinstance(e.__context__, HTTPError):
                e = e.__context__
                if e.response.status_code == 429:
                    log.error('Got 429 (Too many reqs), stopping early.')
                    return
                else:
                    log.error(
                        '%s (%s) request failed: %s',
                        f'/songs/{song_id}',
                        f'{song_full_title}',
                        str(e),
                    )
                    continue
            raise e

        if not lyrics:
            log.warning(
                '%s (%s) returned empty', f'/songs/{song_id}', f'{song_full_title}'
            )
            if lyrics_complete == 'complete':
                # if no result but lyrics are complete, it's prob an instrumental
                # yield a blank record so we don't search it again
                yield {'id': song_id, 'lyrics': None}
            continue

        yield {'id': song_id, 'lyrics': lyrics}


@task
def get_lyrics_task():
    """
    Get lyrics of newly added songs.
    Write results to `genius.lyrics`

    Returns: number of loaded records
    """
    log = logging.getLogger('airflow.task')
    pipeline = dlt.pipeline(
        'genius_lyrics', destination=dlt.destinations.postgres(), dataset_name='genius'
    )
    songs_to_search = sql_table(
        dlt.secrets['sources.postgres.credentials'],
        table='songs_no_lyrics',
        schema='genius',
        backend='connectorx',
        backend_kwargs={'conn': dlt.secrets['sources.postgres.credentials']},
    )
    pipeline.run(
        songs_to_search | get_lyrics,
        table_name='lyrics',
        loader_file_format='parquet',
        primary_key='id',
    )
    row_counts = pipeline.last_trace.last_normalize_info.row_counts or {}
    log.info(f'row counts: {row_counts}')
    return row_counts.get('lyrics', 0)
