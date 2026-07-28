from math import ceil

import pendulum as pnd
from airflow.sdk import CronDataIntervalTimetable, dag

from data_loader.big5 import big5_predict_task
from data_loader.embeddings import (
    MAX_EMBED_LYRICS_JOBS,
    EmbeddingTask,
    make_embed_task_group,
)
from data_loader.genius import (
    genius_search_task,
    get_lyrics_task,
    get_song_metadata_task,
    match_search_results_task,
    match_to_dataset_task,
)
from data_loader.lastfm import get_scrobbles


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
    n_jobs = ceil(MAX_EMBED_LYRICS_JOBS / len(EMBEDDING_TYPES))
    (
        get_scrobbles()
        >> match_to_dataset_task()
        >> genius_search_task()
        >> match_search_results_task()
        >> get_song_metadata_task()
        >> get_lyrics_task()
        >> (
            [make_embed_task_group(task, n_new_jobs=n_jobs) for task in EMBEDDING_TYPES]
            + [big5_predict_task()]
        )
    )


lyrics_analyzer_lastfm()
