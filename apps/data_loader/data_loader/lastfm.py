import logging

import pendulum as pn
from airflow.sdk import task
from airflow.sdk import get_current_context

import dlt
from dlt.sources.helpers.rest_client import RESTClient
from dlt.sources.helpers.rest_client.paginators import PageNumberPaginator
from dlt.sources.helpers.rest_client.auth import APIKeyAuth

MAX_TRACKS_PER_REQUEST = 1000


@dlt.resource
def scrobbles(
    start: pn.DateTime = None,
    end: pn.DateTime = None,
    limit: int = MAX_TRACKS_PER_REQUEST,
):
    """Get recent scrobbles from the current user"""

    def _dt2ts(dt: pn.DateTime) -> int:
        """Convert datetime to unix epoch (secs)"""
        return int(dt.timestamp())

    lastfm_client = RESTClient(
        base_url=dlt.config['sources.lastfm.api_base'],
        auth=APIKeyAuth(
            name='api_key',
            api_key=dlt.secrets['sources.lastfm.api_key'],
            location='query',
        ),
        paginator=PageNumberPaginator(
            base_page=1,
            page_param='page',
            total_path='recenttracks.@attr.totalPages',
        ),
    )

    params = {
        'user': dlt.secrets['sources.lastfm.user'],
        'format': 'json',
        'method': 'user.getRecentTracks',
        'limit': limit,
    }
    if start:
        params['from'] = _dt2ts(start)
    if end:
        params['to'] = _dt2ts(end)

    res = lastfm_client.get('/', params)
    res.raise_for_status()
    for track in res.json()['recenttracks']['track']:
        yield track


def convert_uts(track):
    """Convert scrobble timestamp (uts) to datetime (dt)"""
    track['dt'] = pn.from_timestamp(int(track['date']['uts']))
    track.pop('date')
    return track


def fix_text_fields(track: dict):
    track['artist__mbid'] = track['artist'].get('mbid')
    track['artist'] = track['artist']['#text']
    track['album__mbid'] = track['album'].get('mbid')
    track['album'] = track['album']['#text']
    track['song'] = track.pop('name')
    del track['streamable']
    return track


def filter_now_playing(track: dict):
    return not track.get('@attr', {}).get('nowplaying', False)


def test_pipeline():
    """Test the API by getting the 3 most recent scrobbles since yesterday"""
    pipeline = dlt.pipeline(
        'scrobbles_test', destination='duckdb', dev_mode=True, dataset_name='lastfm'
    )
    pipe = (
        scrobbles(start=pn.yesterday(), limit=10)
        .add_filter(filter_now_playing)
        .add_map(convert_uts)
        .add_map(fix_text_fields)
    )
    pipeline.run(pipe, table_name='scrobbles')
    return pipeline


@task()
def get_scrobbles():
    log = logging.getLogger('airflow.task')
    context = get_current_context()
    start_time = context['data_interval_start']
    end_time = context['data_interval_end']
    log.debug(f'Running task {context["ti"].run_id}')
    log.info(f'Getting scrobbles from {start_time.date()} to {end_time.date()}')

    # pipeline = dlt.pipeline('scrobbles', destination=dlt.destinations.duckdb(), dataset_name='lastfm')
    pipeline = dlt.pipeline(
        'scrobbles', destination=dlt.destinations.postgres(), dataset_name='lastfm'
    )
    pipeline.run(
        (
            scrobbles(start_time, end_time)
            .add_filter(filter_now_playing)
            .add_map(convert_uts)
            .add_map(fix_text_fields)
        ),
        table_name='scrobbles',
        write_disposition='merge',
        primary_key=('url','dt'),
        loader_file_format='parquet',
    )

    row_counts = pipeline.last_trace.last_normalize_info.row_counts
    log.info(f'row counts: {row_counts}')
    return row_counts.get('scrobbles', 0)


if __name__ == '__main__':
    load_info = test_pipeline()
    print(load_info)
