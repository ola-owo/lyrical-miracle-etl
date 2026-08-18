import logging
from base64 import b64encode
from typing import Any

import dlt
import pendulum as pn
from airflow.sdk import task
from dlt import resource, source, transformer
from dlt.common.configuration import configspec
from dlt.sources.helpers.rest_client.auth import OAuth2ClientCredentials
from dlt.sources.helpers.rest_client.client import RESTClient
from dlt.sources.helpers.rest_client.paginators import JSONResponseCursorPaginator

log = logging.getLogger('dlt')

MAX_TRACKS_PER_REQUEST = 50


@configspec
class OAuth2ClientCredentialsHTTPBasic(OAuth2ClientCredentials):
    """Spotify's Oauth2 protocol for access token requests"""

    def build_access_token_request(self) -> dict[str, Any]:
        authentication = b64encode(
            f'{self.client_id}:{self.client_secret}'.encode()
        ).decode()

        log.info(
            {
                'headers': {
                    'Authorization': f'Basic {authentication}',
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                'data': self.access_token_request_data,
            }
        )

        return {
            'headers': {
                'Authorization': f'Basic {authentication}',
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            'data': self.access_token_request_data,
        }


@source
def spotify_recents(
    start_time: pn.DateTime | None = None, end_time: pn.DateTime | None = None
):
    """
    Get spotify song, artist, and album information
    """

    def _dt2ts(dt: pn.DateTime) -> int:
        """Convert datetime to unix epoch (ms)"""
        return int(dt.timestamp() * 1e3)

    assert (start_time is None) ^ (end_time is None), (
        'Specify start_time or end_time but not both'
    )
    spotify_client = RESTClient(
        base_url=dlt.config['sources.spotify.api_base'],
        auth=OAuth2ClientCredentialsHTTPBasic(
            access_token_url=dlt.config['sources.spotify.token_url'],
            client_id=dlt.secrets['sources.spotify.key'],
            client_secret=dlt.secrets['sources.spotify.secret'],
            access_token_request_data={
                'grant_type': 'authorization_code',
                'code': dlt.secrets['sources.spotify.auth_code'],
                'redirect_uri': 'http://{}:{}'.format(
                    dlt.config['sources.spotify.callback_addr'],
                    dlt.config['sources.spotify.callback_port'],
                ),
            },
        ),
        paginator=JSONResponseCursorPaginator(
            cursor_path='cursors.after',  # read next cursor from response JSON
            cursor_param='after',  # send it as ?after=...
        ),
    )

    @resource(write_disposition='append')
    def recently_played(
        before: int | None = None,
        after: int | None = None,
        limit: int = MAX_TRACKS_PER_REQUEST,
    ):
        assert (before is None) ^ (after is None), (
            'Specify before or after but not both'
        )
        params = {'before': before, 'after': after, 'limit': limit}
        if before is None:
            del params['before']
        if after is None:
            del params['after']
        # tracks = spotify_client.get(
        #     '/me/player/recently-played',
        #     params=params,
        # ).json()['tracks']

        # FIELDS_TO_DROP = ['preview_url']  # deprecated field, API always returns null
        # for track in tracks:
        #     for field in FIELDS_TO_DROP:
        #         track.pop(field, None)
        #     yield track
        for page in spotify_client.paginate(
            '/me/player/recently-played',
            params=params,
        ):
            yield from page

    @transformer(data_from=recently_played, write_disposition='merge', primary_key='id')
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

    @transformer(data_from=recently_played, write_disposition='merge', primary_key='id')
    def albums(track_full: dict):
        album = track_full['album']
        album.pop('available_markets', None)
        if 'id' not in album:
            yield {}
        else:
            yield album

    @transformer(data_from=recently_played, write_disposition='merge', primary_key='id')
    def artists(track_full: dict):
        for artist in track_full['artists']:
            if 'id' not in artist:
                yield {}
            else:
                yield artist

    after = _dt2ts(start_time) if start_time else None
    before = _dt2ts(end_time) if end_time else None
    recently_played_tracks = recently_played(before, after)
    return (
        recently_played_tracks | tracks,
        recently_played_tracks | albums,
        recently_played_tracks | artists,
    )


@task
def recent_spotify_tracks(
    start_time: pn.DateTime | None = None, end_time: pn.DateTime | None = None
) -> bool:
    raise NotImplementedError


if __name__ == '__main__':
    pipeline = dlt.pipeline(
        'spotify', destination='duckdb', dev_mode=True, dataset_name='spotify'
    )
    pipeline.run(spotify_recents(pn.yesterday()))
