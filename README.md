# Lyrical Miracle ETL Pipeline

This is an ETL pipeline that pulls my recent music listening history and processes it
for downstream analyses.

## Pipeline breakdown

1. `lastfm/scrobbles`: Get the past week's scrobbles from Last.fm (weekly partitions)
1. `lastfm/genius_matches_from_dataset`: Match scrobbles to the cached Genius song dataset
1. `lastfm/genius_searches`: Search Genius for the remaining unmatched scrobbles
1. `lastfm/genius_matches_from_search`:
   Fuzzy-match search queries and results from the previous step
1. `genius/songs`: Retrieve full song metadata from Genius
1. `genius/lyrics`: Retrieve song lyrics from Genius
1. `genius/lyrics_embed` / `genius/lyrics_embed_clustering`:
   Convert lyrics to embeddings using Gemini (two embed task types)
1. `genius/lyrics_clusters`: K-means clustering of the lyrics embeddings
1. `genius/lyrics_big5`: Get Big-5 scores of song lyrics using a model hosted on Agent Platform

Ingested data is stored in BigQuery; Dagster run/event metadata is stored in PostgreSQL.

## How to run

The pipeline is orchestrated by [Dagster](https://docs.dagster.io/).
Each stage is an asset in `data_loader/`, assembled in `data_loader/definitions.py`:

- `batch_update` — the weekly schedule over the whole chain (Sunday 00:00 UTC)
- `embed_refresh` — on-demand refresh/extract/submit of the Gemini embedding queues
- `cluster_all` — on-demand full re-cluster (regenerates centroids)

### Local run

```sh
uv sync
uv run dagster dev -m data_loader.definitions
```

`dagster dev` picks up the repo-root `dagster.yaml` and `.env` automatically.

### Docker run

One container runs both the Dagster webserver and daemon.

Build the image:

```sh
docker buildx build -t lyrical-miracle-etl .
docker volume create dagster-logs
```

Start the container:

```sh
./docker-run.sh
```

The `.env` file should contain your Dagster, dlt, and GCP config variables.
For example:

```sh
# dagster (run/event storage):
DAGSTER_POSTGRES_URL=postgresql://user:pass@my.database.server/dagster
GOOGLE_APPLICATION_CREDENTIALS=keys/my-service-acct.json

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

### Cloud Run

`cloudbuild.yml` builds the image.
The Cloud Run service needs `min-instances=1` and no CPU throttling
so the daemon isn't frozen between requests, and `DAGSTER_POSTGRES_URL` set in the environment.
See `dagster.yaml` for the instance config.
