from airflow.sdk import dag
from airflow.sdk import CronDataIntervalTimetable
from data_loader.embeddings import embed_lyrics_extract_jobs, embed_lyrics_submit_jobs


@dag(
    dag_id='embed_lyrics_catchup',
    description='Catch up on lyrics embedding',
    schedule=CronDataIntervalTimetable('@hourly', timezone='UTC'),
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1,
)
def embed_lyrics_catchup():
    embed_lyrics_extract_jobs() >> embed_lyrics_submit_jobs()


embed_lyrics_catchup()
