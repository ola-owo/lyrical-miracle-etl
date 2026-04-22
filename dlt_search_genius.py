"""
Search Genius for songs that match our spotify track list

1: read all spotify tracks (from SPOTIFY_TRACKS_FILE)
2. filter out already-matched songs (from genius.song_matches)
3. filter out songs already searched (from genius.genius_searches)
"""

from pathlib import Path
import logging

import polars as pl
import polars.selectors as cs
import pyarrow as pa
from tqdm import tqdm
import duckdb
import dlt
from dlt import transformer
from dlt.sources.helpers.requests.retry import Client

from lyric_analyzer_base.genius import make_client
from lyric_analyzer_base.genius import _excluded_terms
from lyric_analyzer_base.utils import normalize_song_titles

log = logging.getLogger('dlt')

DATA_DIR = Path('data')
SPOTIFY_DATABASE = DATA_DIR / 'spotify_dlt.duckdb'
SPOTIFY_TRACKS_FILE = DATA_DIR / 'tracks.parquet'
SPOTIFY_TRACKS_SAMPLE_FILE = DATA_DIR / 'tracks-sample.parquet'

# Build the lyricsgenius client
excluded_terms = _excluded_terms.copy()
excluded_terms.remove('instrumental')
genius = make_client(skip_non_songs=False, excluded_terms=excluded_terms)
# EXPERIMENTAL: replace lyricsgenius requests session with dlt's session
# https://dlthub.com/docs/api_reference/dlt/sources/helpers/requests/session
genius_headers = genius._session.headers
# genius._session = Session(timeout=10)
# genius._session.headers = genius_headers
genius._session = Client(
    request_timeout=10,
    respect_retry_after_header=True,
    session_attrs=dict(headers=genius_headers),
)


def strip_dlt_fields(doc: dict) -> dict:
    """Strip auto-generated dlt load fields"""
    keys: list[str] = list(doc.keys())
    for k in keys:
        if k.startswith('_dlt_'):
            doc.pop(k)
    return doc


def strip_dlt_fields_arrow(tbl: pa.Table) -> dict:
    """Strip auto-generated dlt load fields"""
    return pl.from_arrow(tbl).drop(cs.starts_with('_dlt_')).to_arrow()


@transformer
def genius_searches(tracks):
    """
    Search Genius for the given song

    Input: spotify track data (dataframe)
    """

    def parse_genius_search_res(hits) -> pl.DataFrame:
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
                (pl.col('lyrics_state') == 'complete').alias('lyrics_complete'),
            )
            .select(cs.all().name.prefix('g_'))
        )

    # remove old DLT load info
    tracks = pl.from_dicts(tracks).drop(cs.starts_with('_dlt_'))

    tracks = tracks.with_columns(
        pl.concat_str(
            [normalize_song_titles(pl.col('name')), pl.col('artist')], separator=' '
        ).alias('searchtext')
    )

    track_iter = tqdm(
        tracks.to_dicts(), desc='searching for tracks', leave=False, unit='req'
    )
    for track in track_iter:
        no_result = {
            # blank record indicates no search results
            'id': track['id'],
            'song': track['name'],
            'artist': track['artist'],
            'searchtext': track['searchtext'],
        }
        res = genius.search_songs(track['searchtext'], per_page=5)
        search_results = parse_genius_search_res(res['hits'])
        if search_results is None:
            yield no_result
            continue
        yield search_results.with_columns(
            pl.lit(track['id']).alias('id'),
            pl.lit(track['name']).alias('song'),
            pl.lit(track['artist']).alias('artist'),
            pl.lit(track['searchtext']).alias('searchtext'),
        ).to_dicts()


# TEST PIPELINE
# duckdb_dest = dlt.destinations.duckdb(DATA_DIR / 'spotify_tests.duckdb')
# pipeline = dlt.pipeline('search_genius', destination=duckdb_dest, dataset_name='genius', dev_mode=True)
# load_info = pipeline.run(
#     (
#         filesystem(f'file://{SPOTIFY_TRACKS_SAMPLE_FILE.absolute()}', file_glob='').add_limit(1)
#         | read_parquet(chunksize=10, use_pyarrow=True).add_map(strip_dlt_fields_arrow)
#         | genius_searches
#     ),
#     table_name='genius_searches',
#     write_disposition='replace',
# )


# FULL PIPELINE
duckdb_dest = dlt.destinations.duckdb(str(SPOTIFY_DATABASE))
pipeline = dlt.pipeline('search_genius', destination=duckdb_dest, dataset_name='genius')

## FILTER OUT TRACKS
# get old searches
if 'genius_searches' in pipeline.dataset().tables:
    old_searches = pl.DataFrame(
        pipeline.dataset()['genius_searches'][['id']].arrow()
    ).unique(cs.all())
    log.info(f'got {old_searches.height} already searched songs')
else:
    old_searches = pl.DataFrame(schema={'id': pl.String})

# get old song matches
with duckdb.connect(SPOTIFY_DATABASE, read_only=True) as cxn:
    try:
        old_matches = cxn.table('genius.song_matches').select('id').pl()
        log.info(f'got {old_matches.height} already matched songs')
    except (duckdb.CatalogException, duckdb.BinderException):
        old_matches = pl.DataFrame(schema={'id': pl.String})

# get list of all ids to skip
ids_to_skip = pl.concat((old_searches, old_matches)).unique(cs.all())
print(f'filtering out {ids_to_skip.height} old searches')

# run the pipeline
CHUNK_SIZE = 100
tracks_df_iter = (
    pl.scan_parquet(SPOTIFY_TRACKS_FILE)
    .drop(cs.starts_with('_dlt_'))
    .join(ids_to_skip.lazy(), on='id', how='anti')
    .collect_batches(CHUNK_SIZE)
)
for tracks_df in tqdm(
    tracks_df_iter,
    desc=f'batch processing',
    leave=False,
    unit='batch',
):
    load_info = pipeline.run(
        tracks_df.to_arrow() | genius_searches(),
        table_name='genius_searches',
        write_disposition='append',
    )

# single pipeline run
# this doesn't work bc if there's an error, the destination isn't written
# load_info = pipeline.run(
#     (
#         filesystem(f'file://{SPOTIFY_TRACKS_FILE.absolute()}', file_glob='')
#         | read_parquet(chunksize=10, use_pyarrow=True)
#             .add_filter()
#             .add_map(strip_dlt_fields)
#         | genius_searches()
#     ),
#     table_name='genius_searches',
#     write_disposition='append',
# )
