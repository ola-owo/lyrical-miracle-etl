"""Dagster definitions for the lyrical-miracle ETL pipeline."""

from math import ceil

import dagster as dg

from data_loader.big5 import big5_ping_asset, lyrics_big5_asset
from data_loader.checks import all_checks
from data_loader.clustering import ClusterConfig, lyrics_clusters_asset
from data_loader.embeddings import (
    MAX_EMBED_JOBS,
    EmbeddingTask,
    make_embed_assets,
)
from data_loader.genius import (
    genius_searches_asset,
    lyrics_asset,
    matches_from_dataset_asset,
    matches_from_search_asset,
    songs_asset,
)
from data_loader.lastfm import scrobbles_asset

EMBED_TYPES: list[EmbeddingTask | None] = [None, EmbeddingTask.CLUSTERING]
N_EMBED_JOBS = ceil(MAX_EMBED_JOBS / len(EMBED_TYPES))

embed_assets: list[dg.AssetsDefinition] = [
    asset
    for embed_type in EMBED_TYPES
    for asset in make_embed_assets(embed_type, n_new_jobs=N_EMBED_JOBS)
]

all_assets: list[dg.AssetsDefinition] = [
    scrobbles_asset,
    matches_from_dataset_asset,
    genius_searches_asset,
    matches_from_search_asset,
    songs_asset,
    lyrics_asset,
    big5_ping_asset,
    lyrics_big5_asset,
    lyrics_clusters_asset,
    *embed_assets,
]

batch_update_job = dg.define_asset_job(
    name='batch_update',
    description='weekly update: get recent songs, get lyrics, and process them',
    selection=dg.AssetSelection.assets(*(asset.key for asset in all_assets)),
)
batch_update_schedule = dg.build_schedule_from_partitioned_job(batch_update_job)

embed_refresh_job = dg.define_asset_job(
    name='embed_refresh',
    description='on-demand refresh the gemini job queue and extract finished jobs',
    selection=dg.AssetSelection.assets(*(asset.key for asset in embed_assets)),
)

cluster_all_job = dg.define_asset_job(
    name='cluster_all',
    description='on-demand re-cluster all lyrics embeddings',
    selection=dg.AssetSelection.assets(lyrics_clusters_asset.key),
    config=dg.RunConfig({'genius__lyrics_clusters': ClusterConfig(replace=True)}),
)

defs = dg.Definitions(
    assets=all_assets,
    asset_checks=all_checks,
    jobs=[batch_update_job, embed_refresh_job, cluster_all_job],
    schedules=[batch_update_schedule],
)
