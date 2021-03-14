import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.evaluate_kicktipp_points import evaluate_kicktipp_432
from utils.slize_games_into_chunks import chunk_games
from utils.plot_gameday_histogram import plot_gameday_hist
from models.heuristic_models import calc_results_kicktipptipper
from load_data import load_odds_and_538_data
from models.models_based_on_betting_odds import create_expected_point_estimation, create_curvefit_model

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)


def evaluate_on_games(df_games):
    df_training_base = df_games.loc[(df_games['season'] <= '2019/2020') | (df_games['league'] != 'D1'), :].copy(
        deep=True)
    df_test_d1 = df_games.loc[(df_games['season'] > '2019/2020') & (df_games['league'] == 'D1'), :].copy(
        deep=True)
    df_training_base.set_index(['league', 'season', 'game_day'], inplace=True, drop=False)
    min_season = df_games.season.min()
    train_split = train_test_split(
        [str(x) for x in df_training_base.loc[df_training_base['season'] > min_season, :].index.unique().values],
        test_size=0.9, random_state=1
    )
    df_test = df_training_base.loc[[str(x) in train_split[1] for x in df_training_base.index], :].append(
        df_test_d1).reset_index(drop=True).sort_values('datetime')
    df_training = df_training_base.loc[[str(x) not in train_split[1] for x in df_training_base.index], :].reset_index(
        drop=True)
    df_test = df_test.loc[df_test['season'] > df_training['season'].min()]
    df_test = df_test.sort_values('datetime')

    max_kt_points_model = create_expected_point_estimation(df_training)
    log_curvefit_model = create_curvefit_model(df_training)
    df_model_results = df_test.copy(deep=True)

    df_model_results['result_max_return'] = [max_kt_points_model(match_odds)
                                             for match_odds in df_model_results['odds_diff']]
    df_model_results['max_exp_points'] = [evaluate_kicktipp_432([home, away], pred[0][0]) for home, away, pred
                                          in zip(df_model_results['FTHG'], df_model_results['FTAG'],
                                                 df_model_results['result_max_return'])]

    df_model_results['result_curve_fit_raw'] = [log_curvefit_model(match_odds)
                                                for match_odds in df_model_results['odds_diff']]
    df_model_results['log_mean'] = [evaluate_kicktipp_432([home, away], np.round(pred)) for home, away, pred
                                    in zip(df_model_results['FTHG'], df_model_results['FTAG'],
                                           df_model_results['result_curve_fit_raw'])]

    df_model_results['result_kicktipper'] = [calc_results_kicktipptipper(odds_home=home, odds_draw=draw, odds_away=away)
                                             for home, draw, away in zip(df_model_results['IWH'],
                                                                         df_model_results['IWD'],
                                                                         df_model_results['IWA'])]
    df_model_results['kttipper'] = [evaluate_kicktipp_432([home, away], pred) for home, away, pred
                                    in zip(df_model_results['FTHG'], df_model_results['FTAG'],
                                           df_model_results['result_kicktipper'])]
    return df_model_results


df_538_eval = load_odds_and_538_data(['D1'])

#######################################
# plot result distribution of seasons #
#######################################
df_agg_by_result = df_538_eval.loc[df_538_eval.league == 'D1', ['result', 'season', 'FTAG']]\
    .groupby(['result', 'season']).count().reset_index().rename(columns={'FTAG': 'occurence'})

df_plot = (df_agg_by_result.loc[df_agg_by_result.season == '2016/2017', ['result', 'occurence']]\
    .set_index('result', drop=True).rename(columns={'occurence': '2016/2017'}) /
           df_538_eval.loc[(df_538_eval.league == 'D1') & (df_538_eval.season == '2016/2017')].shape[0]).join(
    df_agg_by_result.loc[df_agg_by_result.season == '2017/2018', ['result', 'occurence']] \
        .set_index('result', drop=True).rename(columns={'occurence': '2017/2018'}) /
    df_538_eval.loc[(df_538_eval.league == 'D1') & (df_538_eval.season == '2017/2018')].shape[0]
).join(
    df_agg_by_result.loc[df_agg_by_result.season == '2018/2019', ['result', 'occurence']] \
        .set_index('result', drop=True).rename(columns={'occurence': '2018/2019'}) /
    df_538_eval.loc[(df_538_eval.league == 'D1') & (df_538_eval.season == '2018/2019')].shape[0]
).join(
    df_agg_by_result.loc[df_agg_by_result.season == '2019/2020', ['result', 'occurence']] \
        .set_index('result', drop=True).rename(columns={'occurence': '2019/2020'}) /
    df_538_eval.loc[(df_538_eval.league == 'D1') & (df_538_eval.season == '2019/2020')].shape[0]
).join(
    df_agg_by_result.loc[df_agg_by_result.season == '2020/2021', ['result', 'occurence']] \
        .set_index('result', drop=True).rename(columns={'occurence': '2020/2021'}) /
    df_538_eval.loc[(df_538_eval.league == 'D1') & (df_538_eval.season == '2020/2021')].shape[0]
)
df_plot.sort_values(by='2020/2021', ascending=False).plot.bar(figsize=(18, 6))
plt.show()

#########################################
# evaluate different picking strategies #
#########################################
df_results_all_leagues = evaluate_on_games(df_538_eval)

df_gd_all_leagues = df_results_all_leagues.loc[:, ['league', 'season', 'game_day', 'log_mean', 'max_exp_points',
                                                   'kttipper']].groupby(
    ['league', 'season', 'game_day']).sum()
ax1 = plt.subplot(221)
plot_gameday_hist(df_gd_all_leagues, 'kttipper')
plt.subplot(223, sharex=ax1)
plot_gameday_hist(df_gd_all_leagues, 'max_exp_points')
ax2 = plt.subplot(222)
plot_gameday_hist(df_gd_all_leagues, 'log_mean')
plt.show()


df_agg_by_result = df_results_all_leagues.loc[:,
                   ['result', 'log_mean', 'max_exp_points', 'kttipper']
                   ].groupby(['result']).sum()
df_agg_by_result.loc[df_agg_by_result.mean(axis=1) > df_agg_by_result.mean(axis=1).max() / 10, :].plot.bar()
plt.ylabel('Sum of kt-points')
plt.show()

df_agg_by_goal_diff = df_results_all_leagues.loc[
    (df_results_all_leagues['goal_diff'] < 4) & (df_results_all_leagues['goal_diff'] > -4),
    ['goal_diff', 'log_mean', 'max_exp_points', 'kttipper']].groupby(
    ['goal_diff']).mean()
df_agg_by_goal_diff.plot.bar()
plt.ylabel('Average kt-points')
plt.show()

df_chunked = chunk_games(
    df_results_all_leagues.loc[:, ['odds_diff', 'log_mean', 'max_exp_points']],
    chunk_by='odds_diff', chunk_size=100)
df_agg_by_odds_diff = df_chunked.groupby(by='chunk').mean()
df_agg_by_odds_diff['odds_diff'] = df_agg_by_odds_diff.loc[:, ['odds_diff']].round(2)
df_agg_by_odds_diff.set_index('odds_diff', inplace=True)
ax1 = df_agg_by_odds_diff.loc[:, ['log_mean', 'max_exp_points']].plot.bar()
plt.ylabel('Average kt-points')
plt.legend()
plt.show()
