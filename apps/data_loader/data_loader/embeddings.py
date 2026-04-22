from time import sleep
import logging

from airflow.sdk import task

import polars as pl
from google import genai

from lyric_analyzer_base.database import *
from lyric_analyzer_base.keys import get_keys


log = logging.getLogger(__name__)


@task
def embed_lyrics() -> bool:
    '''
    Get song lyrics embeddings using Gemini API

    NOTE: embeddings model (gemini-embedding-001) has a 2048 token limit per request,
    so some long songs might be truncated.

    Embedding model info: ai.google.dev/gemini-api/docs/embeddings#model-versions
    Pricing: ai.google.dev/gemini-api/docs/pricing#batch_13
    '''

    # Number of songs to process per batch
    EMBEDDING_JOB_SIZE = 100
    # recommended embedding sizes are 768, 1536, and 3072 (full size)
    EMBEDDING_DIM = 768
    # wait time in between job status checks
    JOB_POLL_TIME = 60

    # get lyrics that don't have embeddings yet
    df_lyrics_new = duckdb_query(
        'SELECT * FROM "lyrics" '
        'ANTI JOIN "lyrics_embed" USING (g_id) '
        'WHERE lyrics IS NOT NULL',
        read_only=True
    )
    if df_lyrics_new.is_empty():
        log.info('Nothing to do')
        return False

    # build a table of (key, request) pairs to be fed into gemini api
    # we use the genius song id as the key, and the lyrics as the request body
    lyrics_to_obj = lambda text: {"parts": [{"text": text}]}
    struct_type = pl.Struct({
        'parts': pl.List(pl.Struct({
            'text': pl.String
        }))
    })
    lyrics_requests = (
        df_lyrics_new
        .select([
            pl.col('g_id').cast(pl.String).alias('key'),
            pl.col('lyrics').map_elements(lyrics_to_obj, return_dtype=struct_type)
                .alias('request')
        ])
    )
    log.info(f'{lyrics_requests.height} songs need embeddings')

    '''
    - split df_lyrics into blocks
    - convert each block into a list of json requests (`lyrics_request_parts`)
    - package json list into a batch request
    - if request succeeded, download results and append to `embeddings_list`
    '''
    embeddings_list = [] # list of retrieved embeddings
    gemini_client = genai.Client(api_key=get_keys()['gemini_api_key'])
    for df in lyrics_requests.iter_slices(EMBEDDING_JOB_SIZE):
        job = gemini_client.batches.create_embeddings(
            model='gemini-embedding-001',
            src={'inlined_requests': {
                 'config': {'output_dimensionality': EMBEDDING_DIM},
                 'contents': df['request'].to_list()}},
            config={'display_name': 'lyrics-batch-embeddings'})

        while True:
            job = gemini_client.batches.get(name=job.name)
            if job.state.name in ('JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED'):
                break
            log.info(f'Job state: {job.state.name}. Waiting {JOB_POLL_TIME} seconds...')
            sleep(JOB_POLL_TIME)

        if job.state.name == 'JOB_STATE_FAILED':
            log.error(f"Job error: {job.error}")
            break

        log.info(f"Job finished with state: {job.state.name}")
        if job.state.name == 'JOB_STATE_SUCCEEDED':
            resp = job.dest.inlined_embed_content_responses
            embeddings_list.extend([r.response.embedding.values for r in resp])

    # merge embeddings with lyrics (and genius ids),
    df_lyrics_embed = lyrics_requests.select(
        pl.col('key').cast(int).alias('g_id'),
        pl.Series(embeddings_list).cast(pl.Array(pl.Float64, EMBEDDING_DIM)).alias('embedding')
    )

    duckdb_write_table(df_lyrics_embed, 'lyrics_embed', if_exists='append')
    return not df_lyrics_embed.is_empty()
