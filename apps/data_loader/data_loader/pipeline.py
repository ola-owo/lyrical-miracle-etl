from airflow.sdk import dag
import pendulum as pnd
from airflow.sdk import CronDataIntervalTimetable

from lastfm import get_scrobbles
from genius import (
    match_to_dataset_task,
    genius_search_task,
    match_search_results_task,
    get_song_metadata_task,
    get_lyrics_task
)
from embeddings import embed_lyrics_task


@dag(
    dag_id='lyrics_analyzer_lastfm',
    start_date=pnd.datetime(2026, 1, 25, tz='UTC'),
    schedule=CronDataIntervalTimetable('@weekly', timezone='UTC'),
    catchup=True,
    max_active_runs=1,
    max_active_tasks=1, # limited for now bc of duckdb
)
def pipeline():

    (
        get_scrobbles()
        >> match_to_dataset_task()
        >> genius_search_task()
        >> match_search_results_task()
        >> get_song_metadata_task()
        >> get_lyrics_task()
        >> embed_lyrics_task()
    )


pipeline()
