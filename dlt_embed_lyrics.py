"""
Get song metadata from Genius API
"""

import logging
from time import sleep
from pathlib import Path
from math import ceil
from tempfile import NamedTemporaryFile
from io import StringIO

import duckdb
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
from google import genai
from google.genai import types
from tqdm import tqdm

import dlt
from dlt import transformer
from dlt.sources.helpers.requests.retry import Client
from dlt.sources.helpers.requests import HTTPError

log = logging.getLogger('dlt')

JOB_SIZE = 100  # max batch size allowed by gemini api


@transformer()
def embed_lyrics_batch_file(lyrics: pa.Table) -> pa.Table:
    """
    Get batch embeddings from Gemini using the file uploader

    **NOTE:** embeddings model (gemini-embedding-001) has a 2048 token limit per request,
    so some long songs might be truncated.

    - [Embedding model info](https://ai.google.dev/gemini-api/docs/embeddings#model-versions)
    - [Pricing](https://ai.google.dev/gemini-api/docs/pricing#batch_13)
    - [Token counting](https://ai.google.dev/gemini-api/docs/tokens)

    :param lyrics: Table containing Genius song ids and lyrics
    :type lyrics: pa.Table
    :return: Table containing Genius song ids and lyrics embeddings
    :rtype: pa.Table
    """

    # recommended embedding sizes are 768, 1536, and 3072 (full size)
    EMBEDDING_DIM = 768
    # wait time in between job status checks
    JOB_POLL_TIME = 60

    # remove nulls
    n_nulls = pc.sum(lyrics['lyrics'].is_null()).as_py()
    if n_nulls == lyrics.num_rows:
        log.warning('All entries are null. Exiting!')
        return
    elif n_nulls > 0:
        log.warning(f'Removing {n_nulls} null entries')
        lyrics = pc.drop_null(lyrics)

    # make requests jsonl file
    wrap_requests = lambda text: pl.struct(
        output_dimensionality=pl.lit(EMBEDDING_DIM),
        content=pl.struct(parts=pl.concat_list(pl.struct(text=text))),
    )
    requests_df = pl.LazyFrame(lyrics).select(
        key=pl.col('id'), request=wrap_requests(pl.col('lyrics'))
    )

    # Upload the batch file
    gemini_client = genai.Client(api_key=dlt.secrets['sources.gemini.api_key'])
    with NamedTemporaryFile('w+t', prefix='lyrics-batch', suffix='.njdson') as fp:
        requests_df.sink_ndjson(fp.name)
        batch_upload = gemini_client.files.upload(
            file=fp.name,
            config=types.UploadFileConfig(
                display_name='my-batch-requests', mime_type='jsonl'
            ),
        )

    # submit job
    job = gemini_client.batches.create_embeddings(
        model='gemini-embedding-001',
        src={'file_name': batch_upload.name},
        config={'display_name': 'lyrics-batch-embedding-file'},
    )
    log.info(f'Submitted {lyrics.num_rows} requests to job: {job.name}')

    # wait for job to finish
    def poll_job_state(job_name):
        while True:
            yield gemini_client.batches.get(name=job_name)

    for job in tqdm(
        poll_job_state(job.name),
        'waiting for batch job',
        leave=True,
        unit='poll',
    ):
        if job.state.name == 'JOB_STATE_SUCCEEDED':
            log.info(f'Job {job.name} SUCCEEDED')
            break
        elif job.state.name == 'JOB_STATE_CANCELLED':
            log.warning(f'Job {job.name} CANCELLED')
            return
        elif job.state.name == 'JOB_STATE_FAILED':
            log.error(f'Job {job.name} FAILED:')
            log.error(job.error)
            return
        else:
            # log.info(f'{job.state.name}: Waiting {JOB_POLL_TIME} seconds...')
            sleep(JOB_POLL_TIME)

    # build output table
    results_file = job.dest.file_name
    log.info('Batch results file:', results_file)
    with StringIO(
        gemini_client.files.download(file=results_file).decode('utf-8')
    ) as fp:
        results_df = (
            pl.scan_ndjson(fp)
            .select(
                id=pl.col('key'),
                embedding=pl.col('response')
                .struct['embedding']
                .struct['values']
                .cast(pl.Array(pl.Float64, EMBEDDING_DIM)),
            )
            .collect()
        )

    return results_df.to_arrow()


@transformer()
def embed_lyrics(lyrics: pa.Table) -> pa.Table:
    """
    Get batch embeddings from Gemini using inlined requests

    **NOTE:** embeddings model (gemini-embedding-001) has a 2048 token limit per request,
    so some long songs might be truncated.

    - [Embedding model info](https://ai.google.dev/gemini-api/docs/embeddings#model-versions)
    - [Pricing](https://ai.google.dev/gemini-api/docs/pricing#batch_13)
    - [Token counting](https://ai.google.dev/gemini-api/docs/tokens)
    - [Rate limits](https://ai.google.dev/gemini-api/docs/rate-limits#batch-api)

    :param lyrics: Table containing Genius song ids and lyrics
    :type lyrics: pa.Table
    :return: Table containing Genius song ids and lyrics embeddings
    :rtype: pa.Table
    """

    # recommended embedding sizes are 768, 1536, and 3072 (full size)
    EMBEDDING_DIM = 768
    # wait time in between job status checks
    JOB_POLL_TIME = 60

    # remove nulls
    n_nulls = pc.sum(lyrics['lyrics'].is_null()).as_py()
    if n_nulls == lyrics.num_rows:
        log.warning('All entries are null. Exiting!')
        return
    elif n_nulls > 0:
        log.warning(f'Removing {n_nulls} null entries')
        lyrics = pc.drop_null(lyrics)

    # make client and submit job
    gemini_client = genai.Client(api_key=dlt.secrets['sources.gemini.api_key'])
    job = gemini_client.batches.create_embeddings(
        model='gemini-embedding-001',
        src={
            'inlined_requests': {
                'config': {'output_dimensionality': EMBEDDING_DIM},
                'contents': [
                    {'parts': [{'text': text.as_py()}]} for text in lyrics['lyrics']
                ],
            }
        },
        config={'display_name': 'lyrics-batch-embedding-inline'},
    )
    log.info(f'Submitted {lyrics.num_rows} requests to job: {job.name}')

    # wait for job to finish
    def poll_job_state(job_name):
        while True:
            yield gemini_client.batches.get(name=job_name)

    for job in tqdm(
        poll_job_state(job.name),
        'waiting for batch job',
        leave=True,
        unit='poll',
    ):
        if job.state.name == 'JOB_STATE_SUCCEEDED':
            log.info(f'Job {job.name} SUCCEEDED')
            break
        elif job.state.name == 'JOB_STATE_CANCELLED':
            log.warning(f'Job {job.name} CANCELLED')
            return
        elif job.state.name == 'JOB_STATE_FAILED':
            log.error(f'Job {job.name} FAILED:')
            log.error(job.error)
            return
        else:
            # log.info(f'{job.state.name}: Waiting {JOB_POLL_TIME} seconds...')
            sleep(JOB_POLL_TIME)

    # build output table
    resp = job.dest.inlined_embed_content_responses
    embeddings = [r.response.embedding.values for r in resp]

    return pa.Table.from_arrays(
        [
            lyrics['id'],
            pa.array(embeddings, type=pa.list_(pa.float64(), EMBEDDING_DIM)),
        ],
        names=['id', 'embedding'],
    )


SPOTIFY_DB = Path('data/spotify_dlt.duckdb')
DEST_TABLE = 'lyrics_embed'
duckdb_dest = dlt.destinations.duckdb(str(SPOTIFY_DB))
pipeline = dlt.pipeline('lyrics_embed', destination=duckdb_dest, dataset_name='genius')

# get ids of old songs
if DEST_TABLE in pipeline.dataset().tables:
    song_ids_old = pl.DataFrame(pipeline.dataset()[DEST_TABLE][['id']].arrow())
    log.warning(f'filtering out {song_ids_old.height} old songs')
else:
    song_ids_old = pl.DataFrame(schema={'id': pl.Int64})

# get ids of songs to process, excluding old songs
with duckdb.connect(SPOTIFY_DB, read_only=True) as cxn:
    song_ids = cxn.sql("""select id, lyrics from genius.lyrics
                       anti join "song_ids_old" using (id)
                       where lyrics is not null""").pl()
log.info(f'{song_ids.height} songs to search')

song_ids_iter = tqdm(
    song_ids.iter_slices(JOB_SIZE),
    total=ceil(song_ids.height / JOB_SIZE),
    desc=f'batch processing',
    leave=False,
    unit='batch',
)
for song_ids_part in song_ids_iter:
    load_info = pipeline.run(
        song_ids_part.to_arrow() | embed_lyrics(),
        table_name=DEST_TABLE,
        write_disposition='append',
        primary_key='id',
    )
