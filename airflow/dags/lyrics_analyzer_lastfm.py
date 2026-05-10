from airflow.sdk import dag
import pendulum as pnd
from airflow.sdk import CronDataIntervalTimetable

from data_loader.lastfm import get_scrobbles
from data_loader.genius import (
    match_to_dataset_task,
    genius_search_task,
    match_search_results_task,
    get_song_metadata_task,
    get_lyrics_task,
)
from data_loader.embeddings import (
    EmbeddingTask,
    make_embed_task_group,
)


@dag(
    dag_id='lyrics_analyzer_lastfm',
    description='The main pipeline: Get scrobbles, get song lyrics, embed lyrics',
    start_date=pnd.datetime(2026, 1, 25, tz='UTC'),
    schedule=CronDataIntervalTimetable('@weekly', timezone='UTC'),
    catchup=True,
    max_active_runs=1,
)
def lyrics_analyzer_lastfm():
    EMBEDDING_TYPES = (None, EmbeddingTask.CLUSTERING)
    (
        get_scrobbles()
        >> match_to_dataset_task()
        >> genius_search_task()
        >> match_search_results_task()
        >> get_song_metadata_task()
        >> get_lyrics_task()
        >> [make_embed_task_group(task) for task in EMBEDDING_TYPES]
    )


lyrics_analyzer_lastfm()
