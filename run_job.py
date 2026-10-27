#!/usr/bin/env -S uv run --script
"""entrypoint script that runs a specific job"""

from argparse import ArgumentParser
from logging import getLogger

from dagster import DagsterInstance

from data_loader.definitions import defs

parser = ArgumentParser()
parser.add_argument('job')
job_name = parser.parse_args().job

log = getLogger(__name__)

log.info(f'selected job: {job_name}')
job = defs.get_job_def(job_name)

if partitions_def := job.partitions_def:
    partition_key = partitions_def.get_last_partition_key()
    log.info(f'running last partition: {partition_key}')
else:
    partition_key = None
    log.info('running without partitions')

with DagsterInstance.get() as instance:
    result = job.execute_in_process(
        partition_key=partition_key,
        instance=instance,
    )
    assert result.success
