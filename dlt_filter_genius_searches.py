from pathlib import Path

import duckdb
import polars as pl
import polars.selectors as cs
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from lyric_analyzer_base.utils import normalize_song_titles

SPOTIFY_DB = Path('data/spotify_dlt.duckdb')

with duckdb.connect(SPOTIFY_DB, read_only=True) as cxn:
    df_search_results = cxn.sql('select * from genius.genius_searches').pl()
    df_song_matches_old = cxn.sql('SELECT id FROM genius.song_matches').pl()

# Get search results, with "stripped" versions of artist/song/title fields
df_search_results = (
    df_search_results.lazy()
    .drop(cs.starts_with('_dlt'))
    .drop_nulls('g_id')
    .rename(
        {
            'g_primary_artist_names': 'g_artist',
            'g_title': 'g_song',
            'g_full_title': 'g_full_title',
        }
    )
    .with_columns(cs.string().replace(None, ''))
    .with_columns(full_title=pl.col('song') + ' by ' + pl.col('artist'))
    .with_columns(
        normalize_song_titles(
            pl.col('full_title', 'song', 'artist', 'g_full_title', 'g_song', 'g_artist')
        ).name.suffix('_strip')
    )
    .collect()
)

# vectorize song titles/names/artists
titles_mat = TfidfVectorizer(strip_accents='unicode').fit(
    pl.concat((df_search_results['full_title'], df_search_results['g_full_title']))
)
titles_strip_mat = TfidfVectorizer(strip_accents='unicode').fit(
    pl.concat(
        (df_search_results['full_title_strip'], df_search_results['g_full_title_strip'])
    )
)
artists_mat = TfidfVectorizer(strip_accents='unicode').fit(
    pl.concat((df_search_results['artist_strip'], df_search_results['g_artist_strip']))
)
songs_mat = TfidfVectorizer(strip_accents='unicode').fit(
    pl.concat((df_search_results['song_strip'], df_search_results['g_song_strip']))
)


# Remove already-matches songs and score the rest according to text similarity
def text_similarities(df: pl.DataFrame) -> pl.DataFrame:
    """Compute text similarities between songs, artists, titles, and unstripped titles"""
    TEXT_COMPARE_FIELDS = {
        ('title', titles_mat, 'full_title', 'g_full_title'),
        ('title_strip', titles_strip_mat, 'full_title_strip', 'g_full_title_strip'),
        ('artist', artists_mat, 'artist_strip', 'g_artist_strip'),
        ('song', songs_mat, 'song_strip', 'g_song_strip'),
    }
    cosine_sims = {
        name: np.diag(
            cosine_similarity(tfidf.transform(df[c1]), tfidf.transform(df[c2]))
        )
        for (name, tfidf, c1, c2) in TEXT_COMPARE_FIELDS
    }
    return df.with_columns(pl.Series('cos_' + k, v) for k, v in cosine_sims.items())


df_search_results_scored = (
    df_search_results.join(df_song_matches_old, on='id', how='anti')
    .group_by('id')
    .map_groups(text_similarities)
)


# Get the best result from each search
COSINE_CUTOFF = 0.5  # 0.6 retains about 96% of matches
df_search_results_filtered = (
    df_search_results_scored.lazy()
    .with_columns(
        (
            # 0.5 * (pl.col('cos_artist') + pl.col('cos_song')) * pl.col('cos_title')
            0.25 * (pl.col('cos_artist') + pl.col('cos_song'))
            + 0.25 * (pl.col('cos_title') + pl.col('cos_title_strip'))
        ).alias('cos')
    )
    .filter(
        pl.col('cos') > COSINE_CUTOFF, pl.col('cos') == pl.col('cos').max().over('id')
    )
    .unique(['id'])  # if there are multiple top matches, use the 1st one
    .sort('cos')
    .rename({'cos': 'match_score'})
    .collect()
)

# (TESTING) write full data for manual inspection
# df_search_results_filtered.select([
#     'match_score',
#     'full_title', 'g_full_title',
#     'full_title_strip', 'g_full_title_strip',
#     'artist_strip', 'g_artist_strip',
#     'song', 'g_song',
# ]).write_csv('test_search_matches.csv', include_header=True)

# write matches to database
df_song_matches = df_search_results_filtered.select(
    [
        'id',
        'full_title',
        'g_id',
        'g_full_title',
        'match_score',
    ]
)
print('new song matches:')
print(df_song_matches)
with duckdb.connect(SPOTIFY_DB) as cxn:
    cxn.sql('INSERT OR IGNORE INTO genius.song_matches SELECT * FROM "df_song_matches"')
