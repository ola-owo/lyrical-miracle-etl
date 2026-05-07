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
from data_loader.embeddings import embed_lyrics_extract_jobs, embed_lyrics_submit_jobs


@dag(
    dag_id='lyrics_analyzer_lastfm',
    description='The main pipeline: Get scrobbles, get song lyrics, embed lyrics',
    start_date=pnd.datetime(2026, 1, 25, tz='UTC'),
    schedule=CronDataIntervalTimetable('@weekly', timezone='UTC'),
    catchup=True,
    max_active_runs=1,
    max_active_tasks=1,
)
def lyrics_analyzer_lastfm():
    (
        get_scrobbles()
        >> match_to_dataset_task()
        >> genius_search_task()
        >> match_search_results_task()
        >> get_song_metadata_task()
        >> get_lyrics_task()
        >> embed_lyrics_extract_jobs()
        >> embed_lyrics_submit_jobs()
    )


lyrics_analyzer_lastfm()
