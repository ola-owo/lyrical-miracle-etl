"""
Convert Spotify streaming data dump to Parquet files.
More info here: https://support.spotify.com/us/article/understanding-your-data/
"""

import dlt
from dlt.sources.filesystem import filesystem
from dlt.common.typing import TDataItems
from dlt.common.storages.fsspec_filesystem import FileItemDict
import polars as pl
import polars.selectors as cs

from pathlib import Path
from typing import Iterator

SPOTIFY_DATA_DIR = Path('/run/media/ola/T5linux/Data/spotify')
OUT_DIR = SPOTIFY_DATA_DIR


@dlt.transformer
def read_json(files: Iterator[FileItemDict]) -> Iterator[TDataItems]:
    # import duckdb
    import json

    for file_item in files:
        # file_path = file_item.local_file_path
        # print(f'file path: {file_path}')
        # with duckdb.connect() as cxn:
        #     yield cxn.sql(f'''select * from read_json('{file_path}')''').pl()
        with file_item.open('rb') as f:
            yield json.load(f)


fs_resource = filesystem(
    f'file://{SPOTIFY_DATA_DIR}/Spotify_Extended_Streaming_History',
    file_glob='Streaming_History_Audio*.json',
)
pipeline = dlt.pipeline('spotify', destination='duckdb')
load_info = pipeline.run(fs_resource | read_json(), dataset_name='spotify')

# Save all streams
(
    pl.LazyFrame(pipeline.dataset().spotify.arrow()).sink_parquet(
        OUT_DIR / 'Streaming_History_Song.parquet'
    )
)

# Save song plays
(
    pl.LazyFrame(pipeline.dataset().spotify.arrow())
    # drop podcast-specific rows and columns
    .remove(pl.col('spotify_track_uri').is_null())
    .drop(cs.contains('episode'))
    .sink_parquet(OUT_DIR / 'Streaming_History_Song.parquet')
)

# Save podcast plays
(
    pl.LazyFrame(pipeline.dataset().spotify.arrow())
    # drop song-specific rows and cols
    .remove(pl.col('episode_name').is_null())
    .drop(cs.starts_with('master_'))
    .sink_parquet(OUT_DIR / 'Streaming_History_Song.parquet')
)
