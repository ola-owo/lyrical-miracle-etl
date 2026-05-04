"""
Get song metadata from Genius API
"""

import logging
from pathlib import Path
from math import ceil

import duckdb
import polars as pl
import pyarrow as pa
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
def lyrics(songs: pa.Table):
    """
    Scrape song lyrics from Genius
    """
    songs_iter = tqdm(songs.to_pylist(), desc='getting lyrics', leave=False, unit='req')
    for song in songs_iter:
        song_id = song.get('id')
        try:
            lyrics = genius.lyrics(
                song_url=song.get('path'), song_id=song_id, remove_section_headers=True
            )
        except AssertionError as e:
            if isinstance(e.__context__, HTTPError):
                e = e.__context__
                if e.response.status_code != 429:
                    log.error("Request '%s' failed: %s", f'{song_id}', str(e))
                    continue
            raise e

        if not lyrics:
            log.error("Request '%s' returned empty", f'/songs/{song_id}')
            continue

        yield {'id': song_id, 'lyrics': lyrics}


DEST_TABLE = 'lyrics'
duckdb_dest = dlt.destinations.duckdb(str(SPOTIFY_DB))
pipeline = dlt.pipeline('genius_lyrics', destination=duckdb_dest, dataset_name='genius')
if DEST_TABLE in pipeline.dataset().tables:
    song_ids_old = pl.DataFrame(pipeline.dataset()[DEST_TABLE][['id']].arrow())
    log.warning(f'filtering out {song_ids_old.height} old songs')
else:
    song_ids_old = pl.DataFrame(schema={'id': pl.Int64})

with duckdb.connect(SPOTIFY_DB, read_only=True) as cxn:
    song_ids = cxn.sql(
        'SELECT id, path FROM genius.songs'
        ' ANTI JOIN "song_ids_old" USING (id)'
    ).pl()

n_search = song_ids.height
log.info(f'{n_search} songs to search')

CHUNK_SIZE = 100
song_ids_iter = tqdm(
    song_ids.iter_slices(CHUNK_SIZE),
    total=ceil(song_ids.height / CHUNK_SIZE),
    desc=f'batch processing',
    leave=False,
    unit='batch',
)
for song_ids_part in song_ids_iter:
    load_info = pipeline.run(
        song_ids_part.to_arrow() | lyrics(),
        table_name=DEST_TABLE,
        write_disposition='append',
        primary_key='id',
    )
