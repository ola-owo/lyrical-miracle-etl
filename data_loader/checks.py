"""Asset checks guarding data-quality invariants in BigQuery."""

import dagster as dg
import dlt
from google.cloud import bigquery

from data_loader.big5 import BIG5_TRAITS_SHORT
from data_loader.clustering import CENTROIDS_TABLE, CLUSTERS_TABLE
from data_loader.embeddings import EMBEDDING_DIM, EMBEDDING_TABLES, EmbeddingTask

DB_SCHEMA = 'genius'


def _query_count(query: str) -> int:
    """Run a query returning a single count and return that count."""
    client = bigquery.Client(project=dlt.secrets['gcloud.project_id'])
    return client.query_and_wait(query).to_arrow().column(0)[0].as_py()


def make_embedding_dim_check(
    embed_task: EmbeddingTask | None,
) -> dg.AssetChecksDefinition:
    """
    Build a check that all embeddings in a table have the expected dimension.

    Args:
        embed_task: embedding type whose table should be checked

    Returns: asset check definition for that table
    """
    table = EMBEDDING_TABLES[embed_task]

    @dg.asset_check(
        asset=dg.AssetKey(['genius', table]),
        description=f'All embeddings in `{DB_SCHEMA}.{table}` have {EMBEDDING_DIM} dimensions.',
    )
    def embeddings_have_expected_dim() -> dg.AssetCheckResult:
        n_bad = _query_count(
            f'SELECT COUNT(*) FROM {DB_SCHEMA}.{table} '
            f'WHERE ARRAY_LENGTH(embedding) != {EMBEDDING_DIM}'
        )
        return dg.AssetCheckResult(
            passed=n_bad == 0,
            metadata={
                'table': table,
                'rows_with_wrong_dim': n_bad,
                'expected_dim': EMBEDDING_DIM,
            },
        )

    return embeddings_have_expected_dim


@dg.asset_check(
    asset=dg.AssetKey(['genius', CLUSTERS_TABLE]),
    description=(
        'Every cluster label in `lyrics_clusters` has a matching row in '
        '`lyrics_cluster_centroids`.'
    ),
)
def cluster_labels_have_centroids() -> dg.AssetCheckResult:
    n_orphans = _query_count(
        f'SELECT COUNT(*) FROM {DB_SCHEMA}.{CLUSTERS_TABLE} c '
        f'LEFT JOIN {DB_SCHEMA}.{CENTROIDS_TABLE} ct USING (cluster) '
        f'WHERE ct.cluster IS NULL'
    )
    return dg.AssetCheckResult(
        passed=n_orphans == 0,
        metadata={'unmatched_labels': n_orphans},
    )


@dg.asset_check(
    asset=dg.AssetKey(['genius', 'lyrics_big5']),
    description='Every Big5 prediction has all five trait scores.',
)
def big5_traits_are_complete() -> dg.AssetCheckResult:
    null_traits = ' OR '.join(f'{trait} IS NULL' for trait in BIG5_TRAITS_SHORT)
    n_incomplete = _query_count(
        f'SELECT COUNT(*) FROM {DB_SCHEMA}.lyrics_big5 WHERE {null_traits}'
    )
    return dg.AssetCheckResult(
        passed=n_incomplete == 0,
        metadata={'incomplete_rows': n_incomplete},
    )


all_checks: list[dg.AssetChecksDefinition] = [
    make_embedding_dim_check(None),
    make_embedding_dim_check(EmbeddingTask.CLUSTERING),
    cluster_labels_have_centroids,
    big5_traits_are_complete,
]
