from airflow.sdk import dag

from data_loader.embeddings import (
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
    [make_embed_task_group(task) for task in EMBEDDING_TYPES]


embed_lyrics_catchup()
