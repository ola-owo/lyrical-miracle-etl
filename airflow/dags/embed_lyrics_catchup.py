from math import ceil

from airflow.sdk import dag

from data_loader.embeddings import (
    MAX_EMBED_LYRICS_JOBS,
    EmbeddingTask,
    make_embed_task_group,
)


@dag(
    dag_id='embed_lyrics_catchup',
    description='Catch up on lyrics embedding',
    max_active_runs=1,
)
def embed_lyrics_catchup():
    EMBEDDING_TYPES = (None, EmbeddingTask.CLUSTERING)
    n_jobs = ceil(MAX_EMBED_LYRICS_JOBS / len(EMBEDDING_TYPES))
    [make_embed_task_group(task, n_new_jobs=n_jobs) for task in EMBEDDING_TYPES]


embed_lyrics_catchup()
