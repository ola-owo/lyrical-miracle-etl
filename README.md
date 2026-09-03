# Lyrical Miracle ETL Pipeline

This is an ETL pipeline that pulls my recent music listening history and processes it
for downstream analyses.

## Pipeline breakdown

1. `get_scrobbles`: Get the past week's scrobbles from LastFM
1. `match_to_dataset`: Match scrobbles to the cached Genius song dataset
1. `genius_search`: Search Genius for the remaining unmatched scrobbles
1. `match_search_results`: Fuzzy-match search queries and results from the previous step
1. `get_lyrics`: Retrieve song lyrics from Genius
1. `embed_lyrics`: Convert lyrics to embeddings using Gemini
1. `big5`: Get Big-5 scores of song lyrics using a custom model hosted on Agent Platform

Results of each task in are saved to a postgresql database.

## How to run

The entire pipeline can be run in a single Docker container based on the official
[Airflow community image](https://airflow.apache.org/docs/docker-stack/index.html).

Build the image:

```sh
docker build -t lyric-analyzer-etl .
docker volume create airflow-logs
```

Start the container:

```sh
docker run --rm -it -p 8080:8080 \
  --env-file .env \
  --mount type=volume,src=airflow-logs,dst=/opt/airflow/logs \
  --mount type=bind,src=$(pwd)/data,dst=/data,ro \
  lyrical-miracle-etl standalone
```

The `.env` file should contain your Airflow config variables, dlt configs, and dlt secrets.
For example:

```sh
# airflow:
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql://airflow:airflow@my.database.server/airflow_metadata

# dlt:
RUNTIME__LOG_LEVEL=WARNING
DESTINATION__POSTGRES__CREDENTIALS=postgresql://user:pass@my.database.server/lyrics
SOURCES__POSTGRES__CREDENTIALS=postgresql://user:pass@my.database.server/lyrics
SOURCES__LASTFM__API_KEY=...
SOURCES__LASTFM__USER=...
SOURCES__GENIUS__TOKEN=...
SOURCES__GEMINI__API_KEY=...
```

API keys in `.env` are detected by dlt's
[credential mechanism](https://dlthub.com/docs/general-usage/credentials).
When running locally, you'd normally place all your keys in `.dlt/secrets.toml`.
But with Docker, it's easier to pass them as environment variables at runtime.
