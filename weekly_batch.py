#!/usr/bin/env -S uv run --script
"""entrypoint script for weeklyimport dagster as dg batch_update job"""

from dagster import DagsterInstance

from data_loader.definitions import defs

job = defs.get_job_def('batch_update')
if partitions_def := job.partitions_def:
    partition_key = partitions_def.get_last_partition_key()
assert partition_key is not None

with DagsterInstance.get() as instance:
    result = job.execute_in_process(
        partition_key=partition_key,
        instance=instance,
    )
    assert result.success
