import logging
from math import ceil

import dlt
import polars as pl
from airflow.sdk import task
from dlt import transformer
from dlt.common.libs.pyarrow import rename_columns
from dlt.sources.sql_database import sql_table
from google.cloud import aiplatform
from tqdm import tqdm

from data_loader.dlt_utils import get_normalize_row_counts

# global vars
REQUEST_BATCH_SIZE = 8
DB_SCHEMA = 'genius'
BIG5_TRAITS_SHORT = ['OPN', 'CON', 'EXT', 'AGR', 'NEU']


def _check_predict_inputs(texts: pl.DataFrame):
    log = logging.getLogger(__name__)
    n_null = texts['content'].is_null().sum()
    if n_null > 0:
        log.error(f'Batch input has {n_null} null value(s)')
        raise ValueError('Null batch inputs detected')


@transformer
def big5_predict_online(
    texts: pl.DataFrame, endpoint: aiplatform.Endpoint | None = None
):
    """
    :param texts: input table with columns `id` and `content`
    :param dim: embedding size
    """
    log = logging.getLogger('dlt')
    texts_df = pl.DataFrame(texts)
    if texts_df.height == 0:
        log.info('Nothing to embed, exiting.')
        return
    _check_predict_inputs(texts_df)

    if not endpoint:
        log.debug('Initializing endpoint...')
        aiplatform.init(
            project=dlt.secrets['gcloud.project_id'],
            location=dlt.secrets['gcloud.region'],
        )
        endpoint = aiplatform.Endpoint(
            'projects/{}/locations/{}/endpoints/{}'.format(
                dlt.secrets['gcloud.project_id'],
                dlt.secrets['gcloud.region'],
                dlt.secrets['gcloud.big5_endpoint'],
            )
        )

    params = {'include_text': False, 'include_traits': False}
    n_batches = ceil(texts_df.height / REQUEST_BATCH_SIZE)
    log.info(f'Splitting {texts_df.height} reqs into {n_batches} batches')
    for batch in tqdm(
        texts_df.iter_slices(REQUEST_BATCH_SIZE),
        total=n_batches,
        desc='Sending BIG5 inference requests',
        unit='req',
        disable=None,
    ):
        log.debug('Sending inference request...')
        # TODO: add a try/except and catch 429 "Model is not yet ready for inference"
        # if this happens, sleep(?) and retry
        resp = endpoint.predict(instances=batch['content'].to_list(), parameters=params)
        log.debug('Got inference response.')
        preds = pl.DataFrame(
            resp.predictions, schema=BIG5_TRAITS_SHORT, orient='row'
        ).select(batch['id'], pl.all())
        yield preds.to_dicts()


@task
def big5_predict_task() -> dict[str, int]:
    """
    Get new big5 predictions using online inference
    """
    log = logging.getLogger('airflow.task')
    LYRICS_TABLE = 'lyrics_no_big5'

    lyrics_to_embed = sql_table(
        dlt.secrets['sources.bigquery.credentials'],
        table=LYRICS_TABLE,
        schema=DB_SCHEMA,
        included_columns=['id', 'lyrics'],
        backend='pyarrow',
    )
    pipeline = dlt.pipeline(
        'big5',
        dataset_name=DB_SCHEMA,
        destination=dlt.destinations.bigquery(),
    )
    pipeline.run(
        lyrics_to_embed.add_map(lambda t: rename_columns(t, ['id', 'content']))
        | big5_predict_online(),
        table_name='lyrics_big5',
        primary_key='id',
        write_disposition='merge',
        loader_file_format='parquet',
    )
    row_counts = get_normalize_row_counts(pipeline)
    log.info(f'row counts: {row_counts}')
    return row_counts


if __name__ == '__main__':
    big5_predict_task.function()
