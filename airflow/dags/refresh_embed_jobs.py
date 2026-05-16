from airflow.sdk import dag
from airflow.sdk import CronDataIntervalTimetable

from data_loader.embeddings import (
    EmbeddingTask,
    embed_lyrics_refresh_task,
)


@dag(
    dag_id='refresh_embed_jobs',
    description='(TESTING) refresh embed-lyrics jobs table',
    catchup=False,
    max_active_runs=1,
)
def refresh_embed_jobs():
    EMBEDDING_TYPES = (None, EmbeddingTask.CLUSTERING)
    [embed_lyrics_refresh_task(task) for task in EMBEDDING_TYPES]


refresh_embed_jobs()
