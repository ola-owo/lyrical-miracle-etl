"""Partition definitions for the Dagster pipeline."""

import dagster as dg

# Weekly windows starting Sunday 00:00 UTC, aligned with the historical Airflow
# `CronDataIntervalTimetable('@weekly')` data intervals: 2026-01-25 is a Sunday
# and dagster weekly partitions default to Sunday (day_offset=0).
weekly_partitions = dg.WeeklyPartitionsDefinition(
    start_date='2026-01-25',
    timezone='UTC',
)
