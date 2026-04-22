"""
Get song metadata from Genius API
"""

import logging
from pathlib import Path

import duckdb
import polars as pl
from tqdm import tqdm

import dlt
from dlt import transformer
from dlt.sources.helpers.requests.retry import Client
from dlt.sources.helpers.requests import HTTPError

from lyric_analyzer_base.genius import make_client, _excluded_terms
from lyric_analyzer_base.database import *


log = logging.getLogger('dlt')
SPOTIFY_DB = Path('data/spotify_dlt.duckdb')

# Build the lyricsgenius client
excluded_terms = _excluded_terms.copy()
excluded_terms.remove('instrumental')
genius = make_client(skip_non_songs=False, excluded_terms=excluded_terms)
# EXPERIMENTAL: replace lyricsgenius requests session with dlt's session
# https://dlthub.com/docs/api_reference/dlt/sources/helpers/requests/session
genius_headers = genius._session.headers
genius._session = Client(
    request_timeout=10,
    respect_retry_after_header=True,
    session_attrs=dict(headers=genius_headers),
)


@transformer()
def song_meta(song_id: int):
    """
    Lookup song metadata from Genius.
    This uses the public API which has a 10k requests/day limit
    """
    try:
        res = genius.song(song_id)
    except AssertionError as e:
        if isinstance(e.__context__, HTTPError):
            e = e.__context__
            if e.response.status_code != 429:
                log.error("Request '%s' failed: %s", f'/songs/{song_id}', str(e))
                return None
        raise e
    if not res:
        log.error("Request '%s' returned empty", f'/songs/{song_id}')
        return None
    return res['song']


DEST_TABLE = 'songs'
duckdb_dest = dlt.destinations.duckdb(str(SPOTIFY_DB))
pipeline = dlt.pipeline(
    'genius_song_meta', destination=duckdb_dest, dataset_name='genius', progress='tqdm'
)
if DEST_TABLE in pipeline.dataset().tables:
    song_ids_old = pl.DataFrame(pipeline.dataset()[DEST_TABLE][['id']].arrow())
    log.warning(f'filtering out {song_ids_old.height} old songs')
else:
    song_ids_old = pl.DataFrame(schema={'id': pl.Int64})

with duckdb.connect(SPOTIFY_DB, read_only=True) as cxn:
    song_ids = cxn.sql(
        'SELECT DISTINCT g_id AS id FROM genius.song_matches t1'
        ' ANTI JOIN "song_ids_old" t2 ON (t1.g_id = t2.id)'
    ).pl()

n_search = song_ids.height
log.warning(f'{n_search} songs to search')

CHUNK_SIZE = 100
song_ids_iter = tqdm(
    song_ids.iter_slices(CHUNK_SIZE),
    total=round(song_ids.height / CHUNK_SIZE),
    desc=f'getting song metadata ({CHUNK_SIZE}/chunk)',
    leave=False,
    unit='chunk',
)
for song_ids_part in song_ids_iter:
    song_ids_part = song_ids_part['id'].to_arrow()
    load_info = pipeline.run(
        song_ids_part | song_meta(),
        table_name=DEST_TABLE,
        write_disposition='append',
        primary_key='id',
    )
