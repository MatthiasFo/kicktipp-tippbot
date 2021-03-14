import warnings

import numpy as np

warnings.filterwarnings('error')


def chunk_games(df_games, chunk_by, chunk_size=200):
    if chunk_size > (df_games.shape[0] / 6):
        chunk_size = int(df_games.shape[0] / 6)
    df_sorted = df_games.sort_values(by=chunk_by).reset_index()

    num_games = df_sorted.shape[0]
    num_chunks = int(num_games / chunk_size)
    chunks = np.repeat(np.linspace(0, num_chunks - 1, num_chunks), chunk_size)
    df_sorted.loc[df_sorted.index[0:len(chunks)], 'chunk'] = chunks
    # check how many entries remain and put them in another chunk or the last one
    if sum(df_sorted['chunk'].isna()) > chunk_size / 3:
        df_sorted['chunk'].fillna(num_chunks, inplace=True)
    else:
        df_sorted['chunk'].fillna(num_chunks - 1, inplace=True)
    return df_sorted


