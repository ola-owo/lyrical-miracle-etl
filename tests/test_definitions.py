"""Smoke tests for the Dagster definitions."""

from data_loader.definitions import defs

EXPECTED_ASSET_KEYS = {
    'lastfm/scrobbles',
    'lastfm/genius_matches_from_dataset',
    'lastfm/genius_searches',
    'lastfm/genius_matches_from_search',
    'genius/songs',
    'genius/lyrics',
    'genius/big5_ping',
    'genius/lyrics_big5',
    'genius/lyrics_clusters',
    'genius/lyrics_embed',
    'genius/lyrics_embed_jobs_refresh',
    'genius/lyrics_embed_jobs_submit',
    'genius/lyrics_embed_clustering',
    'genius/lyrics_embed_clustering_jobs_refresh',
    'genius/lyrics_embed_clustering_jobs_submit',
}


def test_expected_assets_are_defined():
    keys = {k.to_user_string() for k in defs.resolve_asset_graph().get_all_asset_keys()}
    assert EXPECTED_ASSET_KEYS <= keys


def test_jobs_are_defined():
    for job_name in ['batch_update', 'embed_refresh', 'cluster_all']:
        assert defs.resolve_job_def(job_name).name == job_name


def test_schedule_is_defined():
    schedule = defs.resolve_schedule_def('batch_update_schedule')
    assert schedule.job_name == 'batch_update'
