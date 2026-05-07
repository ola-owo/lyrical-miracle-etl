import logging
from time import sleep
import json
from enum import StrEnum
from io import StringIO
from typing import Iterable
from pathlib import Path

import pyarrow as pa
import polars as pl
from tqdm import tqdm
import fsspec
import duckdb
from adbc_driver_postgresql import dbapi

from google import genai
from google.genai import Client
from google.genai.types import BatchJob, JobState
from google.genai.errors import ClientError

from airflow.sdk import task
import dlt
from dlt import transformer, resource
from dlt.sources.sql_database import sql_table

from data_loader.dlt_utils import get_normalize_row_counts


# gemini api batch limits
MAX_JOB_SIZE = 100
MAX_CONCURRENT_JOBS = 100

# recommended embedding sizes are 768, 1536, and 3072 (full size)
EMBEDDING_DIM = 768

# Airflow task limits
MAX_EMBED_LYRICS_JOBS = 8


class TaskTypesV1(StrEnum):
    """gemini-embedding-001 task types"""

    SEMANTIC_SIMILARITY = 'SEMANTIC_SIMILARITY'
    CLASSIFICATION = 'CLASSIFICATION'
    CLUSTERING = 'CLUSTERING'
    RETRIEVAL_DOCUMENT = 'RETRIEVAL_DOCUMENT'
    RETRIEVAL_QUERY = 'RETRIEVAL_QUERY'
    CODE_RETRIEVAL_QUERY = 'CODE_RETRIEVAL_QUERY'
    QUESTION_ANSWERING = 'QUESTION_ANSWERING'
    FACT_VERIFICATION = 'FACT_VERIFICATION'


class TaskTypesV2(StrEnum):
    """gemini-embedding-2 task types"""

    # asymmetric:
    SEARCH_QUERY = 'search result'
    QUESTION_ANSWERING = 'question answering'
    FACT_CHECKING = 'fact checking'
    CODE_RETRIEVAL = 'code retrieval'
    # symmetric:
    CLASSIFICATION = 'classification'
    CLUSTERING = 'clustering'
    SEMANTIC_SIMILARITY = 'sentence similarity'


def _make_client(api_key=dlt.secrets['sources.gemini.api_key']) -> Client:
    """Get a Gemini API client"""
    return genai.Client(api_key=api_key)


def _check_batch_inputs(texts: pl.DataFrame):
    log = logging.getLogger(__name__)
    if texts.height > MAX_JOB_SIZE:
        log.error(f'Batch size {texts.height} is over the limit of {MAX_JOB_SIZE}')
        raise ValueError('Batch is too large!')
    n_null = texts['content'].is_null().sum()
    if n_null > 0:
        log.error(f'Batch input has {n_null} null value(s)')
        raise ValueError('Null batch inputs detected')


def _poll_job_state(job_name: str, client: Client):
    while True:
        yield client.batches.get(name=job_name)


def _wait_for_job(job_name: str, client: Client, wait_time=60) -> BatchJob:
    """Wait for batch job to finish, then return the finished job"""
    log = logging.getLogger(__name__)
    for job in tqdm(
        _poll_job_state(job_name, client),
        'waiting for batch job',
        leave=True,
        unit=' polls',
    ):
        if job.state.name in ('JOB_STATE_PENDING', 'JOB_STATE_RUNNING'):
            log.debug(f'Job {job.name}: {job.state.name}, waiting {wait_time} secs...')
            sleep(wait_time)
        elif job.state.name == 'JOB_STATE_SUCCEEDED':
            log.info(f'Job {job.name} SUCCEEDED')
            break
        elif job.state.name == 'JOB_STATE_CANCELLED':
            log.warning(f'Job {job.name} CANCELLED')
            break
        elif job.state.name == 'JOB_STATE_FAILED':
            log.error(f'Job {job.name} FAILED:')
            log.error(job.error)
            break
        else:
            log.error(f'Job {job.name} unknown state: {job.state.name}')
            break
        return job


@transformer
def batch_embed_inline(
    texts, client: Client = None, disp_name='batch-embedding-inline', dim=EMBEDDING_DIM
):
    """
    Get embeddings of given song lyrics using Gemini batch embedding API,
    using the new `gemini-embedding-2` model.
    Use inline requests and wait for the job to complete.

    Input table columns:
    - `id`: Item ID number
    - `content`: text to embed

    :param texts: input table
    :param client: gemini API client
    :param disp_name: job display name
    :param dim: embedding size

    Returns table (id, embedding)
    """
    log = logging.getLogger('dlt')
    texts = pl.from_arrow(texts)
    if texts.height == 0:
        log.info('Nothing to embed, exiting.')
        return
    _check_batch_inputs(texts)

    client = client or _make_client()
    job = client.batches.create_embeddings(
        model='gemini-embedding-001',
        src={
            'inlined_requests': {
                'config': {'output_dimensionality': dim},
                'contents': [
                    {'parts': [{'text': text.as_py()}]}
                    for text in texts['content'].to_arrow()
                ],
            }
        },
        config={'display_name': disp_name},
    )
    log.info(f'Submitted {texts.height} requests to job: {job.name}')

    # keep polling until job ends
    job = _wait_for_job(job.name, client, 60)
    resp = job.dest.inlined_embed_content_responses
    embeddings = [r.response.embedding.values for r in resp]
    return pa.Table.from_arrays(
        [
            texts['id'],
            pa.array(embeddings, type=pa.list_(pa.float64(), EMBEDDING_DIM)),
        ],
        names=['id', 'embedding'],
    )


@transformer
def submit_batch_file_job(
    texts, client=None, disp_name='batch-embedding-inline', dim=EMBEDDING_DIM
):
    """
    Submit a batch embedding job using Files API and gemini-embedding-2 model

    Input table columns:
    - `id`: item ID number
    - `title`: (optional) document title
    - `content`: text to embed

    :param texts: input table
    :param client: gemini API client
    :param disp_name: job display name
    :param dim: embedding size

    :return: the submitted job
    """
    log = logging.getLogger(__name__)
    # log = logging.getLogger('dlt')
    texts = pl.from_arrow(texts)
    if texts.height == 0:
        log.warning('Nothing to embed, exiting.')
        return
    _check_batch_inputs(texts)

    texts = (
        texts.with_columns(title=pl.coalesce('^title$', pl.lit('none')))
        .with_columns(pl.col('title', 'content').str.replace_all('|', '', literal=True))
        .with_columns(prompt=pl.format('title: {} | text: {}', 'title', 'content'))
    )

    client = client or _make_client()

    # create file with embedding requests,
    # then upload it to gemini files api
    TEMP_FILE = 'memory:///batch.jsonl'
    reqs = (
        {
            'key': r['id'],
            'request': {
                'output_dimensionality': dim,
                'content': {'parts': [{'text': r['prompt']}]},
            },
        }
        for r in texts.select('id', 'prompt').iter_rows(named=True)
    )
    with fsspec.open(TEMP_FILE, 'wt') as f:
        for req in reqs:
            json.dump(req, f)
            f.write('\n')
    with fsspec.open(TEMP_FILE, 'rb') as f:
        reqs_file = client.files.upload(file=f, config={'mimeType': 'jsonl'})
    log.debug(f'Uploaded request content to {reqs_file.name}')
    try:
        job = client.batches.create_embeddings(
            model='gemini-embedding-2',
            src={'file_name': reqs_file.name},
            config={'display_name': disp_name},
        )
    except ClientError as e:
        log.error(f'Gemini client error: {e}')
        return
    log.info(f'Submitted {texts.height} requests to job: {job.name}')
    job_dict = job.__dict__.copy()
    job_dict.pop('dest', None)
    return job_dict


def _parse_batch_file_job(job_name: str, client: Client = None, delete_finished=False):
    """
    Check a background batch embeddings job.
    If it's finished, return the embeddings as a table (id, embedding)
    If not finished, return None

    :param job_name: name of the job to check
    :param gemini: Gemini API client
    :param delete_finished: delete completed job after extracting embeddings (default True)

    :returns: yields `pyarrow.Table` with columns `(id, embedding)`
    """
    log = logging.getLogger(__name__)
    client = client or _make_client()

    job = client.batches.get(name=job_name)
    if job.state == JobState.JOB_STATE_SUCCEEDED:
        log.info(f'Job {job.name} completed')
        if not job.dest:
            log.error('Job destination is blank!')
            return
        if job.dest.inlined_embed_content_responses and not job.dest.file_name:
            log.warning('This is an inline job, not file-based (skipping)')
            return
        res = client.files.download(file=job.dest.file_name)
        emb = pl.read_ndjson(StringIO(res.decode())).select(
            id=pl.col('key'),
            embedding=pl.col('response')
            .struct['embedding']
            .struct['values']
            .cast(pl.List(pl.Float64())),
        )
        yield emb.to_arrow()
    elif job.state in (JobState.JOB_STATE_PENDING, JobState.JOB_STATE_RUNNING):
        log.info(f'Job {job.name} in progress... ({job.state.name})')
    elif job.state.name == JobState.JOB_STATE_CANCELLED:
        log.warning(f'Job {job.name} CANCELLED')
    elif job.state.name == JobState.JOB_STATE_FAILED:
        log.error(f'Job {job.name} FAILED:')
        log.error(job.error)
    else:
        log.error(f'Job {job.name} unknown state: {job.state.name}')
        return

    if delete_finished and job.state not in (
        JobState.JOB_STATE_PENDING,
        JobState.JOB_STATE_RUNNING,
    ):
        log.info(f'Deleting job {job.name} from the queue')
        client.batches.delete(name=job.name)


@resource
def retrieve_batch_file_job(job_name: str, client: Client = None, delete_finished=True):
    """
    Check a background batch embeddings job.
    If it's finished, return the embeddings as a table (id, embedding)
    If not finished, return None

    :param job_name: name of the job to retrieve
    :param gemini: Gemini API client
    :param delete_finished: delete completed job after extracting (default True)

    :returns: Table with columns `(id, embedding)`
    """
    client = client or _make_client()
    job = client.batches.get(name=job_name)
    yield from _parse_batch_file_job(job.name, client, delete_finished)


@resource
def retrieve_all_batch_file_jobs(client: Client = None, delete_finished=True):
    """
    Run `batch_embed_file_retrieve` on all jobs in the queue

    :param gemini: Gemini API client
    :param delete_finished: delete completed job after extracting (default True)
    """
    client = client or _make_client()
    for job in client.batches.list():
        yield _parse_batch_file_job(job.name, client, delete_finished)


@resource
def refresh_all_batch_jobs(client: Client = None):
    """
    Refresh the state of all queued jobs

    :param client: Gemini API client
    :returns: Job state, excluding `dest`
    :rtype: dict
    """
    client = client or _make_client()
    for job in client.batches.list():
        job_dict = job.__dict__.copy()
        job_dict.pop('dest', None)
        yield job_dict


@transformer
def batch_embed_v1(
    texts,
    client: Client = None,
    disp_name='batch-embedding-inline',
    dim=EMBEDDING_DIM,
    task: TaskTypesV1 = None,
):
    """
    Get embeddings of given song lyrics using Gemini batch embedding API,
    using the (now deprecated) `gemini-embedding-001` model

    Input table columns:
    - `id`: Item ID number
    - `content`: text to embed

    :param texts: input table
    :param client: gemini API client
    :param disp_name: job display name
    :param dim: embedding size

    Returns table (id, embedding)
    """
    JOB_POLL_TIME = 60  # wait time in seconds between job status checks

    log = logging.getLogger('dlt')
    texts = pl.from_arrow(texts)
    if texts.height == 0:
        log.info('Nothing to embed, exiting.')
        return
    _check_batch_inputs(texts)

    client = client or _make_client()
    job_config = {'display_name': disp_name}
    if task:
        job_config['task_type'] = task
    job = client.batches.create_embeddings(
        model='gemini-embedding-001',
        src={
            'inlined_requests': {
                'config': {'output_dimensionality': dim},
                'contents': [
                    {'parts': [{'text': text.as_py()}]}
                    for text in texts['content'].to_arrow()
                ],
            }
        },
        config=job_config,
    )
    log.info(f'Submitted {texts.height} requests to job: {job.name}')
    job = _wait_for_job(job.name, client, JOB_POLL_TIME)
    resp = job.dest.inlined_embed_content_responses
    embeddings = [r.response.embedding.values for r in resp]
    return pa.Table.from_arrays(
        [
            texts['id'],
            pa.array(embeddings, type=pa.list_(pa.float64(), EMBEDDING_DIM)),
        ],
        names=['id', 'embedding'],
    )


@transformer
def retrieve_batch_inline_job(
    job_name: BatchJob, ids: list[str], client: Client, delete_finished=True
):
    """
    Retrieve embeddings from a batch inline job.
    If the job's not finished, return None

    :param job_name: name of the job to retrieve
    :param ids: ids of batch job queries
    :client: Gemini API client
    :delete_finished: delete completed job after extracting (default True)

    :returns: Table with columns `(id, embedding)`
    """
    log = logging.getLogger('dlt')
    client = client or _make_client()
    job = client.batches.get(name=job_name)  # get full job with output data
    if job.state != JobState.JOB_STATE_SUCCEEDED:
        log.warning(f'Job {job.name} is incomplete: {job.state.name}')
        return
    resp = job.dest.inlined_embed_content_responses
    embeddings = (r.response.embedding.values for r in resp)
    yield pa.Table.from_arrays(
        [
            ids,
            pa.array(embeddings, type=pa.list_(pa.float64(), EMBEDDING_DIM)),
        ],
        names=['id', 'embedding'],
    )
    if delete_finished and job.state not in (
        JobState.JOB_STATE_PENDING,
        JobState.JOB_STATE_RUNNING,
    ):
        log.info(f'Deleting job {job.name} from the queue')
        client.batches.delete(name=job.name)


def _load_normalized_package_to_duckdb(job_files: Iterable[Path], table, schema):
    schema = 'genius'
    table = 'lyrics_embed'
    table_full_name = f'"{schema}"."{table}"'

    load_df = pl.read_parquet(job_files)
    duckdb.register('load_df', load_df)
    with duckdb.connect(dlt.secrets['destination.duckdb.credentials']) as cxn:
        cxn.begin()
        cxn.execute(f'CREATE SCHEMA IF NOT EXISTS {schema}')
        cxn.execute(
            f'CREATE TABLE IF NOT EXISTS {table_full_name} (id bigint primary key, embedding double[])'
        )
        cxn.execute(f'INSERT OR REPLACE INTO {table_full_name} SELECT * FROM "load_df"')
        cxn.commit()  # TODO: check data before committing


def _load_normalized_package_to_pg(
    job_files: Path | Iterable[Path], table, schema, staging_schema=None
):
    """
    Load normalized data into postgres
    Load to a staging table, then upsert into main table

    Args:
        job_files: parquet file(s) to load
        table: name of table to load into
        schema: schema to load into
        staging_schema: schema to use for staging table
    """
    log = logging.getLogger(__name__)
    staging_schema = staging_schema or schema + '_staging'
    table_full = f'"{schema}"."{table}"'
    staging_table_full = f'"{staging_schema}"."{table}"'

    with dbapi.connect(
        dlt.secrets['destination.postgres.credentials'], autocommit=False
    ) as cxn:
        with cxn.cursor() as cur:
            n_ingest = cur.adbc_ingest(
                table,
                pl.scan_parquet(job_files).unique('id').collect().to_arrow(),
                mode='replace',
                db_schema_name=staging_schema,
            )
        cxn.commit()
        log.info(f'Wrote {n_ingest} records to {staging_table_full}')
        with cxn.cursor() as cur:
            cur.execute(
                f'CREATE TABLE IF NOT EXISTS {table_full} (id bigint primary key, embedding float[])'
            )
            merge_res = cur.execute(f"""MERGE INTO {table_full} t
                        USING {staging_table_full} stg ON t.id = stg.id
                        WHEN NOT MATCHED THEN
                        INSERT (id, embedding) VALUES(stg.id, stg.embedding)
                        WHEN MATCHED AND t.embedding != stg.embedding THEN
                        UPDATE SET embedding = stg.embedding
                        RETURNING merge_action(), stg.id""").fetch_polars()
        merge_res = merge_res.group_by('merge_action').agg(pl.len()).rows()
        log.info(f'{table} staging -> main merge: {merge_res}')
        cxn.commit()


def _load_all_normalized(pipeline: dlt.Pipeline, table):
    """
    Load all normalized data into the pipeline's destination

    Args:
        pipeline: dlt pipeline
        table: name of table to load into
    """
    log = logging.getLogger(__name__)
    if pipeline.destination.destination_type == 'dlt.destinations.duckdb':
        load_fn = _load_normalized_package_to_duckdb
    elif pipeline.destination.destination_type == 'dlt.destinations.postgres':
        load_fn = _load_normalized_package_to_pg
    else:
        raise NotImplementedError(
            'Unsupported destination type:', pipeline.destination.destination_type
        )

    load_storage = pipeline._get_load_storage()
    for load_id in load_storage.list_normalized_packages():
        log.info(f'Finding files from load package {load_id}')
        load_pkg_dir = Path(pipeline.working_dir).joinpath(
            'load', 'normalized', load_id, 'new_jobs'
        )
        job_files = sorted(
            f for f in load_pkg_dir.glob('*.parquet') if f.name.startswith(table)
        )
        log.debug(f'Files to load: {job_files}')
        if not load_pkg_dir.exists():
            raise RuntimeError(
                f'No new load jobs found for {pipeline.pipeline_name} job {load_id}'
            )
        if not job_files:
            raise RuntimeError('No normalized parquet files found for `lyrics_embed`')
        load_fn(job_files, table=table, schema=pipeline.dataset_name)
        log.debug(f'Clearing package {load_id} from load queue')
        load_storage.complete_load_package(
            load_id, aborted=False
        )  # remove package from load queue


@task
def embed_lyrics_extract_jobs() -> dict[str, int]:
    """
    Extract embeddings from completed batch jobs.

    Returns:
        row counts for all normalized tables.
    """
    log = logging.getLogger('airflow.task')
    gemini = _make_client()
    n_active_jobs = len(gemini.batches.list()) > 0

    pipeline = dlt.pipeline(
        'embed_lyrics',
        destination=dlt.destinations.postgres(),
        dataset_name='genius',
    )

    if not n_active_jobs:
        log.debug('No active running jobs')
        return {}

    log.info(f'Checking {n_active_jobs} jobs currently in the queue')
    pipeline.extract(
        retrieve_all_batch_file_jobs(gemini, delete_finished=True),
        table_name='lyrics_embed',
        write_disposition='merge',
        primary_key='id',
        loader_file_format='parquet',
    )
    pipeline.normalize()
    _load_all_normalized(pipeline, 'lyrics_embed')
    row_counts = get_normalize_row_counts(pipeline)
    log.info(f'row counts: {row_counts}')
    return row_counts


@task
def embed_lyrics_submit_jobs(n_job_slots=MAX_EMBED_LYRICS_JOBS) -> dict[str, int]:
    """
    Submit new lyrics embedding batch jobs.

    Returns:
        row counts for all normalized tables.
    """
    log = logging.getLogger('airflow.task')
    assert n_job_slots <= MAX_CONCURRENT_JOBS
    gemini = _make_client()
    n_active_jobs = len(gemini.batches.list()) > 0

    pipeline = dlt.pipeline(
        'embed_lyrics',
        destination=dlt.destinations.postgres(),
        dataset_name='genius',
    )

    n_active_jobs = len(gemini.batches.list())
    n_new_jobs = n_job_slots - n_active_jobs
    if n_new_jobs <= 0:
        log.warning('Concurrent jobs limit reached, not submitting any new jobs')
        return {}

    lyrics_to_embed = sql_table(
        dlt.secrets['sources.postgres.credentials'],
        table='lyrics_no_embed',
        schema='genius',
        included_columns=['id', 'lyrics'],
        backend='connectorx',
        chunk_size=MAX_JOB_SIZE,
        backend_kwargs={
            'conn': dlt.secrets['sources.postgres.credentials'],
            'return_type': 'arrow_stream',
        },
    )
    pipeline.run(
        (
            lyrics_to_embed
            .add_limit(n_new_jobs)
            .add_map(lambda t: t.rename_columns(['id', 'content']))
            | submit_batch_file_job(disp_name='lyrics-batch-embed-file')
        ),
        table_name='lyrics_embed_jobs',
        write_disposition='replace',
        primary_key='name',
        loader_file_format='parquet',
    )
    row_counts = get_normalize_row_counts(pipeline)
    log.info(f'row counts: {row_counts}')
    return row_counts
