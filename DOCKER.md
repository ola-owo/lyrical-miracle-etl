# Docker

This image runs Airflow in one container using the same command shape as local
development:

```sh
docker build -t lyric-analyzer-etl .
```

Run with Airflow's default local SQLite metadata database:

```sh
docker run --rm -it -p 8080:8080 lyric-analyzer-etl
```

Run with an external Airflow metadata database:

```sh
docker run --rm -it -p 8080:8080 \
  -e AIRFLOW_DATABASE_SQL_ALCHEMY_CONN='postgresql+psycopg2://airflow:airflow@host.docker.internal:5432/airflow' \
  lyric-analyzer-etl
```

`AIRFLOW_DATABASE_SQL_ALCHEMY_CONN` is a convenience alias. You can also use
Airflow's native setting directly:

```sh
-e AIRFLOW__DATABASE__SQL_ALCHEMY_CONN='postgresql+psycopg2://airflow:airflow@host.docker.internal:5432/airflow'
```

Pass dlt secrets with dlt's environment variable form. For example:

```sh
-e SOURCES__POSTGRES__CREDENTIALS='postgresql://user:password@host.docker.internal:5432/lyrics'
-e SOURCES__LASTFM__API_KEY='...'
-e SOURCES__LASTFM__USER='...'
-e SOURCES__GENIUS__TOKEN='...'
-e SOURCES__GEMINI__API_KEY='...'
```

For persistent Airflow logs and the local SQLite metadata database, mount a
volume at `/opt/airflow` or mount narrower paths such as `/opt/airflow/logs`.
