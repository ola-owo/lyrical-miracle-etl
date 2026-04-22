import os

from typing import Literal
import duckdb
from polars import DataFrame, Series

# sqlite database
# DB_NAME = 'scrobbles.db'
# SQLITE_DB_URI = f'sqlite+pysqlite:///{DB_NAME}' # sqlite and sqlalchemy
# CNX_SQLITE_URI = f'sqlite://{DB_NAME}' # connectorX
# SCROBBLE_TBL = 'scrobbles'
# eng = create_engine(SQLITE_DB_URI)

# postgres (local) database
# CNX_PSG_URI = f'postgresql://postgres:postgres@localhost:5432/discogs'
# PSG_DB_URI = f'postgresql+psycopg2://postgres:postgres@localhost:5432/discogs'
# eng = create_engine(PSG_DB_URI)

DUCKDB_FILE = os.environ.get('DUCKDB_FILE', 'data/spotify_dlt.duckdb')


def duckdb_list_tables(cxn: duckdb.DuckDBPyConnection = None) -> Series:
    """
    List all database tables (and views).

    :param cxn: Duckdb connection
    :type cxn: duckdb.DuckDBPyConnection
    :return: List of table names
    :rtype: Series
    """
    if cxn:
        return cxn.sql('SHOW TABLES').pl().get_column('name')
    with duckdb.connect(DUCKDB_FILE, read_only=True) as cxn:
        return cxn.sql('SHOW TABLES').pl().get_column('name')


def duckdb_read_table(tablename: str) -> DataFrame:
    """
    Read a table (or view) and return its contents as a DataFrame.

    :param tablename: Table name
    :type tablename: str
    """
    with duckdb.connect(DUCKDB_FILE, read_only=True) as cxn:
        return cxn.sql(f'select * from "{tablename}"').pl()


def duckdb_write_table(
    df: DataFrame,
    tablename: str,
    if_exists: Literal['fail', 'append', 'replace', 'upsert'] = 'fail',
):
    """
    Write a dataframe to the database.

    :param df: DataFrame to be written
    :type df: DataFrame
    :param tablename: Table to write to
    :type tablename: str
    :param if_exists: What to do if the table already exists
    :type if_exists: Literal['fail', 'append', 'replace']
    """

    # def filter_df_cols(cxn, df):
    #     return df.select([c for c in cxn.table(tablename).columns if c in df.columns])

    with duckdb.connect(DUCKDB_FILE) as cxn:
        all_tables = duckdb_list_tables(cxn)
        if tablename in all_tables:
            if if_exists == 'fail':
                raise Exception(f'Table "{tablename}" already exists!')
            elif if_exists == 'append':
                # df = filter_df_cols(cxn, df)
                cxn.sql(f'INSERT INTO "{tablename}" BY NAME SELECT * FROM df')
            elif if_exists == 'replace':
                cxn.sql(f'DROP TABLE IF EXISTS "{tablename}"')
                cxn.sql(f'CREATE TABLE "{tablename}" AS SELECT * FROM df')
            elif if_exists == 'upsert':
                df = df.select(
                    [c for c in cxn.table(tablename).columns if c in df.columns]
                )
                cxn.sql(
                    f'INSERT OR REPLACE INTO "{tablename}" BY NAME SELECT * FROM df'
                )
            else:
                raise Exception(f'Unknown parameter: if_exists="{if_exists}"')
        else:
            cxn.sql(f'CREATE TABLE "{tablename}" AS SELECT * FROM df')


def duckdb_merge_table(
    df: DataFrame, tablename: str, on: list[DataFrame], overwrite=False
):
    """
    Merge a dataframe into a duckdb table.

    :param df: polars dataframe to write
    :param tablename: name of database table
    :param on: columns to join on
    :param overwrite: if True, overwrite rows that exist in both tables
    """
    raise Exception('TODO')


def duckdb_query(query: str, params: list = None, read_only=False) -> DataFrame:
    """
    Run a query against the database and return the result as a DataFrame.
    """
    if not params:
        params = []
    with duckdb.connect(DUCKDB_FILE, read_only=read_only) as cxn:
        return cxn.execute(query, params).pl()
