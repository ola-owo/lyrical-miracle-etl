~~~
title: lyrical-miracle-etl
~~~

# Summary

This is an ETL pipeline that pulls my recent LastFM listening history and processes it
for downstream analyses.

# Pipeline description

1. `get_scrobbles`: Get the past week's scrobbles from LastFM
1. `match_to_dataset`: Match scrobbles to the cached Genius song dataset
1. `genius_search`: Search Genius for the remaining unmatched scrobbles
1. `match_search_results`: Fuzzy-match search queries and results from the previous step
1. `get_lyrics`: Retrieve song lyrics from Genius
1. `embed_lyrics`: Convert lyrics to embeddings using Gemini

Results of each task in are saved to a Neon database.

# How to run

The entire pipeline can be run in a single Docker container
based on the official [Airflow community image](https://airflow.apache.org/docs/docker-stack/index.html).

Build the image:

```sh
docker build -t lyric-analyzer-etl .
```

```sh
# Run with Airflow's default local SQLite metadata database:
docker run --rm -it -p 8080:8080 lyric-analyzer-etl

# Run with an external Airflow metadata database:
docker run --rm -it -p 8080:8080 \
  -e AIRFLOW__DATABASE__SQL_ALCHEMY_CONN='MY_DATABASE_URI' \
  lyric-analyzer-etl
```

API keys are handled by dlt's [credential mechanism](https://dlthub.com/docs/general-usage/credentials)
When running locally, you'd normally place all your keys in `.dlt/secrets.toml`.
When running through Docker, it's easier to pass the keys as environment variables when starting the container.
For example:

```sh
# contents of .env:
DESTINATION__POSTGRES__CREDENTIALS=...
SOURCES__POSTGRES__CREDENTIALS=...
SOURCES__LASTFM__API_KEY=...
SOURCES__LASTFM__USER=...
SOURCES__GENIUS__TOKEN=...
SOURCES__GEMINI__API_KEY=...


docker run --env-file .env [...]
```

For persistent Airflow logs and the local SQLite metadata database, mount a
volume at `/opt/airflow` or mount narrower paths such as `/opt/airflow/logs`.
