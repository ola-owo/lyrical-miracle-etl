import io
import logging

import dagster as dg
import dlt
import numpy as np
import polars as pl
from google.cloud import bigquery
from sklearn.cluster import KMeans

from data_loader.embeddings import EMBEDDING_TABLES, EmbeddingTask

EMBEDDING_DIM = 768

# table names
DB_SCHEMA = 'genius'
EMBEDDING_TABLE = EMBEDDING_TABLES[EmbeddingTask.CLUSTERING]
EMBEDDING_UNCLUSTERED_TABLE = 'lyrics_embed_no_clustering'
CLUSTERS_TABLE = 'lyrics_clusters'
CENTROIDS_TABLE = 'lyrics_cluster_centroids'

# clustering variables
N_CLUSTERS = 7
RANDOM_SEED = 1738


def run_kmeans(embeddings, n_clusters=N_CLUSTERS, random_seed=RANDOM_SEED) -> KMeans:
    """
    Run K-means

    Args:
        embeddings: array or polars series containing embeddings (as row vectors)
        n_clusters: number of clusters to fit (ignored if km is given)
        random_seed: random state passed to KMeans (ignored if km is given)

    Returns: KMeans object
    """
    log = logging.getLogger(__name__)
    log.info(f'Running k-means on {len(embeddings)} vectors')
    return KMeans(n_clusters, random_state=random_seed).fit(embeddings)


def build_kmeans_from_centroids(centroids: np.ndarray) -> KMeans:
    """
    Build KMeans from an existing list of centroids

    Args: array containing centroids (as row vectors)

    Returns: KMeans object
    """
    return KMeans(
        n_clusters=centroids.shape[0],
        init=centroids,  # pyright: ignore[reportArgumentType]
        n_init=1,  # pyright: ignore[reportArgumentType]
        max_iter=1,
    ).fit(centroids)


def get_embeddings(fetch_all=False) -> pl.DataFrame:
    """
    Pull lyrics embeddings from bigquery into a dataframe

    Args:
        fetch_all: if true, fetch all embeddings. if false, pull unclustered embeddings only

    Returns: dataframe with columns (id, embedding)
    """
    if fetch_all:
        table = EMBEDDING_TABLE
    else:
        table = EMBEDDING_UNCLUSTERED_TABLE
    log = logging.getLogger(__name__)
    client = bigquery.Client(project=dlt.secrets['gcloud.project_id'])

    table_full = f'{DB_SCHEMA}.{table}'
    if table not in {t.table_id for t in client.list_tables(DB_SCHEMA)}:
        log.warning(f'Table {table_full} not found; returning empty dataframe')
        return pl.DataFrame(
            schema={'id': pl.Int64, 'embedding': pl.Array(pl.Float64, EMBEDDING_DIM)}
        )

    if fetch_all:
        log.info(f'Fetching all embeddings from {table_full}')
    else:
        log.info(f'Fetching unclustered embeddings from {table_full}')
    rows = client.query_and_wait(f'SELECT id, embedding FROM {table_full}')
    embeddings = pl.DataFrame(rows.to_arrow())
    return embeddings.with_columns(
        pl.col('embedding').cast(pl.Array(pl.Float64, EMBEDDING_DIM))
    )


def kmeans_to_dataframe(km, ids, labels) -> dict[str, pl.DataFrame]:
    """
    Convert Kmeans result (km) to polars dataframes

    Args:
        km: KMeans object
        ids: list-like containing ids of embedding vectors
        labels: list-like containing cluster ids

    Returns: dictionary containing cluster labels and centroids

    The first dataframe "labels" has columns:
        id (int): embedding id
        cluster (int): cluster label

    The second dataframe "centroids" has columns:
        cluster (int): cluster number
        centroid (array[double]): centroid vector
    """
    assert len(ids) == len(labels), 'ids and labels have unequal length'
    labels = pl.DataFrame(
        {
            'id': pl.Series('id', ids, dtype=pl.Int64),
            'cluster': pl.Series('cluster', labels, dtype=pl.Int64),
        }
    )
    centroids = (
        pl.DataFrame({'centroid': km.cluster_centers_})
        .with_columns(
            pl.col('centroid').cast(pl.List(pl.Float64)),
            pl.row_index('cluster').cast(pl.Int64),
        )
        .select('cluster', 'centroid')
    )
    return {'labels': labels, 'centroids': centroids}


def _load_parquet_table(df: pl.DataFrame, table_full: str, write_disposition: str):
    """
    Write a polars dataframe to a bigquery table via parquet

    Args:
        df: dataframe to write
        table_full: fully-qualified table name ({dataset}.{table})
        write_disposition: bigquery write disposition (e.g. WRITE_TRUNCATE)
    """
    client = bigquery.Client(project=dlt.secrets['gcloud.project_id'])
    opts = bigquery.format_options.ParquetOptions()  # pyright: ignore[reportAttributeAccessIssue]
    opts.enable_list_inference = True  # convert arrow lists to bq arrays
    buf = io.BytesIO()
    df.write_parquet(buf)
    buf.seek(0)
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        parquet_options=opts,
        write_disposition=write_disposition,
    )
    job = client.load_table_from_file(buf, table_full, job_config=job_config)
    job.result()  # wait for load to finish


def write_clustering(labels, centroids, replace=True) -> dict[str, int]:
    """
    Write clustering results to bigquery.

    Write cluster labels to CLUSTERS_TABLE, and write centroids to CENTROIDS_TABLE.
    Create the table(s) if it doesn't already exist.

    Args:
        labels: dataframe with columns (id, cluster)
        centroids: dataframe with columns (cluster, centroid)
        replace: if true, replace clusters table. if false, upsert.

    Returns: row counts by table
    """
    log = logging.getLogger(__name__)
    centroids_tbl_full = f'{DB_SCHEMA}.{CENTROIDS_TABLE}'
    clusters_tbl_full = f'{DB_SCHEMA}.{CLUSTERS_TABLE}'
    row_counts = {}

    if replace:
        centroids = centroids.select(
            pl.col('cluster').cast(pl.Int64),
            pl.col('centroid').cast(pl.List(pl.Float64)),
        ).sort('cluster')
        _load_parquet_table(
            centroids, centroids_tbl_full, bigquery.WriteDisposition.WRITE_TRUNCATE
        )
        log.info(f'Wrote {centroids.height} centroids to {centroids_tbl_full}')
        row_counts[CENTROIDS_TABLE] = centroids.height

    labels = labels.select(
        pl.col('id').cast(pl.Int64),
        pl.col('cluster').cast(pl.Int64),
    )
    if replace:
        _load_parquet_table(
            labels, clusters_tbl_full, bigquery.WriteDisposition.WRITE_TRUNCATE
        )
        log.info(f'Wrote {labels.height} rows to {clusters_tbl_full} (replace)')
        row_counts[CLUSTERS_TABLE] = labels.height
    else:
        staging_tbl_full = clusters_tbl_full + '_staging'
        _load_parquet_table(
            labels, staging_tbl_full, bigquery.WriteDisposition.WRITE_TRUNCATE
        )
        client = bigquery.Client(project=dlt.secrets['gcloud.project_id'])
        client.query_and_wait(
            f'CREATE TABLE IF NOT EXISTS {clusters_tbl_full} AS '
            f'SELECT * FROM {staging_tbl_full} WHERE false'
        )
        merge_res = client.query_and_wait(f"""
            MERGE {clusters_tbl_full} t
            USING {staging_tbl_full} stg ON t.id = stg.id
            WHEN NOT MATCHED THEN
            INSERT (id, cluster) VALUES(stg.id, stg.cluster)
            WHEN MATCHED THEN
            UPDATE SET cluster = stg.cluster""")
        n_merged = merge_res.num_dml_affected_rows or 0
        log.info(
            f'merged {n_merged} rows from {staging_tbl_full} to {clusters_tbl_full}'
        )
        row_counts[CLUSTERS_TABLE] = n_merged

    return row_counts


def _get_centroids() -> np.ndarray:
    """
    Fetch existing cluster centroids from bigquery

    Returns: array of centroids (as row vectors)
    """
    client = bigquery.Client(project=dlt.secrets['gcloud.project_id'])

    table_full = f'{DB_SCHEMA}.{CENTROIDS_TABLE}'
    if CENTROIDS_TABLE not in {t.table_id for t in client.list_tables(DB_SCHEMA)}:
        raise RuntimeError(f'No centroids found: {table_full} does not exist')

    res = client.query_and_wait(f'SELECT cluster, centroid FROM {table_full}')
    centroids = (
        pl.DataFrame(res.to_arrow())
        .sort('cluster')
        .with_columns(pl.col('centroid').cast(pl.Array(pl.Float64, EMBEDDING_DIM)))
    )
    return centroids['centroid'].to_numpy()


class ClusterConfig(dg.Config):
    """
    Run config for the lyrics clustering asset

    Args:
        replace: cluster all embeddings and generate new centroids (True), or
            incrementally label unclustered embeddings using existing
            centroids (False)
        n_clusters: number of clusters (ignored if replace=False)
        random_seed: random seed passed to KMeans (ignored if replace=False)
    """

    replace: bool = False
    n_clusters: int = N_CLUSTERS
    random_seed: int = RANDOM_SEED


@dg.asset(
    key_prefix='genius',
    name='lyrics_clusters',
    group_name='clustering',
    deps=[dg.AssetKey(['genius', 'lyrics_embed_clustering'])],
    description=(
        'Label lyrics embeddings with K-means clusters. '
        'A full recluster (replace=True) also rewrites `genius.lyrics_cluster_centroids`.'
    ),
)
def lyrics_clusters_asset(
    context: dg.AssetExecutionContext, config: ClusterConfig
) -> dg.MaterializeResult[dict[str, int]]:
    """
    Run KMeans clustering on lyrics embeddings.
    Write cluster labels to `genius.lyrics_clusters`; with replace=True,
    also write new centroids to `genius.lyrics_cluster_centroids`.

    Args:
        replace: if true, cluster all embeddings and generate new centroids.
            if false, use existing centroids and process un-clustered
            embeddings only (requires centroids to exist).

    Returns: row counts by table
    """
    log = context.log
    replace = config.replace

    embeddings = get_embeddings(fetch_all=replace)
    embeddings_arr = embeddings['embedding'].to_numpy()
    if embeddings.height == 0:
        log.warning('No embeddings to cluster, exiting.')
        return dg.MaterializeResult(metadata={'row_counts': {}}, value={})

    if replace:
        km = run_kmeans(
            embeddings_arr,
            n_clusters=config.n_clusters,
            random_seed=config.random_seed,
        )
        km_dfs = kmeans_to_dataframe(km, embeddings['id'], km.labels_)
    else:
        km = build_kmeans_from_centroids(_get_centroids())
        labels = km.predict(embeddings_arr)
        km_dfs = kmeans_to_dataframe(km, embeddings['id'], labels)

    log.info(
        f'Cluster sizes: {km_dfs["labels"].group_by("cluster").agg(pl.len().alias("size")).sort("cluster")}'
    )
    row_counts = write_clustering(
        km_dfs['labels'], km_dfs['centroids'], replace=replace
    )
    log.info(f'row counts: {row_counts}')
    return dg.MaterializeResult(metadata={'row_counts': row_counts}, value=row_counts)
