"""
Get info on all tracks in the spotify data dump
API endpoint: https://developer.spotify.com/documentation/web-api/reference/get-several-tracks
"""

from typing import Iterable
from pathlib import Path
import logging

import polars as pl
import dlt
from dlt import source, resource, transformer
from dlt.sources.helpers.rest_client import RESTClient
from dlt.sources.helpers.rest_client.auth import OAuth2ClientCredentials

log = logging.getLogger('dlt')

SPOTIFY_DATA_DIR = Path('/run/media/ola/T5linux/Data/MUSIC/spotify-dump')
# SPOTIFY_DATA_DIR = Path('data')
SPOTIFY_DATA_FILE = SPOTIFY_DATA_DIR / 'Streaming_History_Song.parquet'
SPOTIFY_MAX_TRACKS_PER_REQUEST = 50

# from dlt.sources.filesystem import filesystem, read_parquet
# fs_resource = filesystem(f'file:/{SPOTIFY_DATA_DIR}', file_glob=SPOTIFY_DATA_FILE.name)
# fs_pipe = (fs_resource | read_parquet()).with_name('spotify')


@source
def spotify(track_ids: Iterable[str]):
    """
    Get spotify song, artist, and album information
    """
    spotify_client = RESTClient(
        base_url=dlt.config['sources.spotify.api_base'],
        auth=OAuth2ClientCredentials(
            access_token_url=dlt.config['sources.spotify.token_url'],
            client_id=dlt.secrets['sources.spotify.key'],
            client_secret=dlt.secrets['sources.spotify.secret'],
        ),
    )

    @resource(write_disposition='append')
    def tracks_fulldata(track_ids: Iterable[str]):
        for track_id in track_ids:
            res = spotify_client.get(f'/tracks/{track_id}')
            log.warning(f"API RESPONSE: {res}")
            track = res.json()
            track.pop('preview_url', None)
            yield track

    @transformer(data_from=tracks_fulldata, write_disposition='merge', primary_key='id')
    def tracks(track: dict):
        if 'id' not in track:
            return

        main_artist = track.get('artists', [{}])[0]
        track['artist'] = main_artist.get('name')
        track['artist_id'] = main_artist.get('id')

        album = track.get('album', {})
        track['album'] = album.get('name')
        track['album_id'] = album.get('id')

        track.pop('artists', None)
        track.pop('album', None)
        track.pop('available_markets', None)

        return track

    @transformer(data_from=tracks_fulldata, write_disposition='merge', primary_key='id')
    def albums(track_full: dict):
        album = track_full['album']
        album.pop('available_markets', None)
        if 'id' not in album:
            yield {}
        else:
            yield album

    @transformer(data_from=tracks_fulldata, write_disposition='merge', primary_key='id')
    def artists(track_full: dict):
        for artist in track_full['artists']:
            if 'id' not in artist:
                yield {}
            else:
                yield artist

    return (
        tracks_fulldata(track_ids) | tracks,
        tracks_fulldata(track_ids) | albums,
        tracks_fulldata(track_ids) | artists,
    )


# TEST PIPELINE
streams = pl.scan_parquet(SPOTIFY_DATA_DIR / 'Streaming_History_Song.parquet')
track_ids_sample = (
    streams.head(100)
    .select(id = pl.col('spotify_track_uri').unique()
            .str.split_exact(':', 2).struct.field('field_2'))
    .collect().to_series()
    .sample(3)
)
pipeline = dlt.pipeline('spotify', destination='duckdb', dev_mode=True)
load_info = pipeline.run(spotify(track_ids_sample))


# FULL PIPELINE
# get existing track ids from duckdb
# pipeline = dlt.pipeline(
#     'spotify',
#     destination=dlt.destinations.duckdb(str(SPOTIFY_DATA_DIR / 'spotify_dlt.duckdb')),
# )
# if 'tracks' in pipeline.dataset('spotify').tables:
#     old_tracks = (
#         pl.LazyFrame(pipeline.dataset('spotify').table('tracks').arrow())
#         .select(pl.col('id').unique())
#         .collect()
#     )
# else:
#     old_tracks = pl.LazyFrame(schema={'id': pl.String})

# track_ids_all = (
#     pl.scan_parquet(SPOTIFY_DATA_FILE)
#     .select(
#         pl.col('spotify_track_uri')
#         .unique()
#         .str.split_exact(':', 2)
#         .struct.field('field_2')
#         .alias('id')
#     )
#     .join(old_tracks.lazy(), on='id', how='anti')  # filter out existing track ids
# )
# track_ids_iter = track_ids_all.collect_batches(
#     chunk_size=SPOTIFY_MAX_TRACKS_PER_REQUEST
# )
# for track_ids in track_ids_iter:
#     load_info = pipeline.run(spotify(track_ids['id']))
#     print(load_info)
