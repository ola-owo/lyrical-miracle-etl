#!/bin/sh
# Run the dagster webserver and daemon in one container
# (the `airflow standalone` replacement).
#
# Expects DAGSTER_HOME to point at the directory containing dagster.yaml,
# and DAGSTER_POSTGRES_URL to be set in the environment.
set -e

: "${PORT:=3000}"
export PATH="/opt/dagster/app/.venv/bin:$PATH"

dagster-daemon run -m data_loader.definitions &
exec dagster-webserver -h 0.0.0.0 -p "$PORT" -m data_loader.definitions
