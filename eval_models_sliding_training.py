import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load_data import load_odds_and_538_data
from models.heuristic_models import calc_results_kicktipptipper
from models.model_dixon_cole_with_spi_odds import create_combined_model
from models.models_based_on_betting_odds import create_curvefit_model
from utils.evaluate_kicktipp_points import evaluate_kicktipp_432
from utils.plot_gameday_histogram import plot_gameday_hist

df_data = load_odds_and_538_data(['D1', 'E0'])
df_data = df_data[['team1', 'team2', 'FTHG', 'FTAG', 'datetime', 'league', 'season', 'game_day',
                   'goal_diff', 'IWH', 'IWA', 'IWD', 'proj_score1', 'proj_score2', 'spi1', 'spi2', 'odds_diff']]
df_data = df_data.dropna(how='all')

league_to_test = 'D1'
df_data.set_index(['league', 'season', 'game_day'], inplace=True, drop=False)
test_indices = df_data.loc[(df_data.season > '2017/2018') &
                           (df_data.league == league_to_test), :].index.unique().values
dict_tests = []
for run_idx in range(len(test_indices)):
    test_idx = test_indices[run_idx]
    print('On run ' + str(run_idx) + ' of ' + str(len(test_indices)))

    df_test_games = df_data.loc[[x == test_idx for x in df_data.index], :]
    league = df_test_games.league[0]
    current_season = int(df_test_games.season[0][:4])
    previous_season = str(current_season - 1) + '/' + str(current_season)
    df_train_games_dc = df_data.loc[(df_data.datetime < df_test_games.datetime.min()) &
                                    (df_data.season >= previous_season) &
                                    (df_data.league == league), :].copy(deep=True)

    df_train_games_538 = df_data.loc[(df_data.datetime < df_test_games.datetime.min()), :].copy(deep=True)
    df_train_games_538.reset_index(drop=True, inplace=True)
    combined_model = create_combined_model(df_train_games_538, 'D1')
    log_curvefit_model = create_curvefit_model(df_train_games_538)

    for index, row in df_test_games.iterrows():
        kttipper_pick = calc_results_kicktipptipper(odds_home=row['IWH'], odds_draw=row['IWD'],
                                                    odds_away=row['IWA'])
        kt_points_kttipper = evaluate_kicktipp_432([row['FTHG'], row['FTAG']], kttipper_pick)

        log_fit_pick = log_curvefit_model(row['IWH'] - row['IWA'])
        kt_points_fit = evaluate_kicktipp_432([row['FTHG'], row['FTAG']], np.round(log_fit_pick))

        inp_data = row[['team1', 'team2', 'IWH', 'IWD', 'IWA', 'spi1', 'spi2', 'proj_score1', 'proj_score2']].values
        combined_pick = combined_model(*inp_data)
        kt_points_comb = evaluate_kicktipp_432([row['FTHG'], row['FTAG']], combined_pick[0][0])

        dict_tests.append({'HomeTeam': row['team1'],
                           'AwayTeam': row['team2'],
                           'FTHG': row['FTHG'],
                           'FTAG': row['FTAG'],
                           'goal_diff': row['goal_diff'],
                           'season': row['season'],
                           'game_day': row['game_day'],
                           'league': row['league'],
                           'kt_points_kttipper': kt_points_kttipper,
                           'kt_points_comb': kt_points_comb,
                           'pick_comb': combined_pick[0][0],
                           'pick_ktt': kttipper_pick,
                           'pick_fit': np.round(log_fit_pick),
                           'kt_points_fit': kt_points_fit})
df_eval = pd.DataFrame(dict_tests)
df_plot = df_eval.loc[:,
          ['league', 'season', 'game_day', 'kt_points_comb', 'kt_points_kttipper', 'kt_points_fit']].groupby(
    ['league', 'season', 'game_day']).sum()
ax1 = plt.subplot(311)
plot_gameday_hist(df_plot, 'kt_points_fit')
plt.subplot(312, sharex=ax1, sharey=ax1)
plot_gameday_hist(df_plot, 'kt_points_comb')
plt.subplot(313, sharex=ax1, sharey=ax1)
plot_gameday_hist(df_plot, 'kt_points_kttipper')
plt.show()

df_agg_by_goal_diff = df_eval.loc[
    (df_eval['goal_diff'] < 4) & (df_eval['goal_diff'] > -4),
    ['goal_diff', 'kt_points_comb', 'kt_points_kttipper', 'kt_points_fit']].groupby(
    ['goal_diff']).sum()
df_agg_by_goal_diff.plot.bar()
plt.ylabel('Sum kt-points')
plt.show()
