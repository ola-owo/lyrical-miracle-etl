#!/bin/sh
set -e
: "${PORT:=3000}"
uv run --no-sync dg dev -m data_loader.definitions -p "$PORT" -h 0.0.0.0
