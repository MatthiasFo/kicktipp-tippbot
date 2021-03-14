import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils.slize_games_into_chunks import chunk_games
from utils.generate_and_convert_picks import get_likely_results, convert_pick_array_to_string
from load_data import load_league_data_with_odds
from models.models_based_on_betting_odds import evaluate_picks

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df_data = load_league_data_with_odds(range(2006, 2021), ['D1', 'E0'])

df_chunked = chunk_games(df_data, chunk_by='odds_diff', chunk_size=500)
df_picked_games = evaluate_picks(df_chunked)
df_expected_points = df_picked_games.groupby(by='chunk').agg(lambda x: np.mean(x) - np.std(x) / 2)
df_mean = df_picked_games.groupby(by='chunk').mean()
df_expected_points['odds_diff'] = df_mean['odds_diff'].round(4)

xdata = [str(round(x, 3)) for x in df_expected_points['odds_diff'].values]
x = np.arange(len(xdata))
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
pick = '2-1'
ax.bar(x - 3 * width / 2, df_expected_points[pick], width, label=pick)
pick = '2-0'
ax.bar(x - width / 2, df_expected_points[pick], width, label=pick)
pick = '1-2'
ax.bar(x + width / 2, df_expected_points[pick], width, label=pick)
ax.set_xticks(x)
ax.set_xticklabels(xdata)
plt.legend()
plt.show()

for chunk_idx in np.linspace(1, df_picked_games['chunk'].max(), 4):
    picks = [convert_pick_array_to_string(x) for x in get_likely_results() if (x[0] < 3) and (x[1] < 3)]
    df_picked_games.loc[df_picked_games['chunk'] == int(chunk_idx), picks].hist()
    fig = plt.gcf()
    fig.suptitle('mean odds diff: ' + str(
        df_picked_games.loc[df_picked_games['chunk'] == int(chunk_idx), 'odds_diff'].mean().round(3)), fontsize=14)
    plt.show()
