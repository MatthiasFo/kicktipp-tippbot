import numpy as np
from scipy.optimize import curve_fit

from models.basic_math_functions import non_neg_logistic
from utils.evaluate_kicktipp_points import evaluate_kicktipp_432_vectorized
from utils.generate_and_convert_picks import get_likely_results, convert_pick_array_to_string, \
    convert_pick_string_to_array
from utils.slize_games_into_chunks import chunk_games


def create_curvefit_model(df_games):
    popt_log_home, _ = curve_fit(non_neg_logistic, df_games['odds_diff'], df_games['FTHG'])
    popt_log_away, _ = curve_fit(non_neg_logistic, df_games['odds_diff'], df_games['FTAG'])

    def estimate_with_logistic_curve(odds_diff):
        home_score = non_neg_logistic(odds_diff, *popt_log_home)
        away_score = non_neg_logistic(odds_diff, *popt_log_away)
        return [home_score, away_score]

    return estimate_with_logistic_curve


def evaluate_picks(df_games):
    likely_picks = get_likely_results()
    for pick in likely_picks:
        df_games[convert_pick_array_to_string(pick)] = \
            evaluate_kicktipp_432_vectorized(df_games[['FTHG', 'FTAG']].values, pick)
    return df_games


def create_expected_point_estimation(df_input):
    df_working_copy = df_input.copy(deep=True)

    def estimate_percentile(x):
        cut_off = 0.01
        conf = np.mean(x) - 0 * np.std(x) / 2
        return conf if conf > cut_off else cut_off

    def estimate_ktpoints_37percentile(df_games, feature_columns=None):
        if feature_columns is None:
            feature_columns = ['odds_diff']

        df_games = chunk_games(df_games, chunk_size=20, chunk_by=feature_columns)
        df_expect_points = df_games.groupby(by='chunk').agg(lambda x: estimate_percentile(x))
        df_mean = df_games.groupby(by='chunk').mean()
        df_expect_points[feature_columns] = df_mean[feature_columns]
        return df_expect_points

    df_picked_games = evaluate_picks(df_working_copy)
    df_expect_points = estimate_ktpoints_37percentile(df_picked_games)
    picks = [convert_pick_array_to_string(x) for x in get_likely_results()]

    pick_dict = {}
    num_chunks = df_expect_points.index.max()
    for chunk_idx in range(int(num_chunks)):
        df_iter = df_expect_points.loc[chunk_idx, picks]
        mean_odds = df_expect_points.loc[chunk_idx, 'odds_diff']
        sorted_picks = sorted([[convert_pick_string_to_array(a), b] for a, b
                               in zip(df_iter.index, df_iter.values)], key=lambda l: l[1], reverse=True)
        pick_dict[mean_odds] = sorted_picks

    def pick_with_expected_points(x):
        key_list = [[abs(x - y), y] for y in pick_dict.keys()]
        idx_key = np.argmin([y[0] for y in key_list])
        return pick_dict[key_list[idx_key][1]]

    return pick_with_expected_points
