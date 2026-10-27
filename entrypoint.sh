#!/bin/bash
set -e
: "${PORT:=3000}"
if (( $# == 0 )); then
	uv run dg dev -m data_loader.definitions -p "$PORT" -h 0.0.0.0
else
	uv run -s run_job.py "$@"
fi
