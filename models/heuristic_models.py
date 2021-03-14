import math
import numpy as np
from scipy.optimize import curve_fit

from models.basic_math_functions import straight_line


# has a horrible performance :-(
def create_odds_probability_model(df_input):
    df_data = df_input.copy(deep=True)
    df_data['est_home_prob'] = [1 / x for x in df_data['IWH'].values]
    df_data['est_away_prob'] = [1 / x for x in df_data['IWA'].values]
    df_data['est_draw_prob'] = [1 / x for x in df_data['IWD'].values]

    df_data['est_home_away_diff'] = df_data['est_home_prob'] - df_data['est_away_prob']

    df_data['is_home_win'] = df_data['FTHG'] > df_data['FTAG']
    df_data['is_away_win'] = df_data['FTHG'] < df_data['FTAG']
    df_data['is_draw'] = df_data['FTHG'] > df_data['FTAG']

    popt_home, _ = curve_fit(straight_line, df_data['est_home_prob'], df_data['is_home_win'])
    pop_diff, _ = curve_fit(straight_line, df_data['est_home_away_diff'], df_data['goal_diff'])
    popt_away, _ = curve_fit(straight_line, df_data['est_away_prob'], df_data['is_away_win'])
    popt_draw, _ = curve_fit(straight_line, df_data['est_draw_prob'], df_data['is_draw'])

    def pick_by_odds_prob(odds_home, odds_draw, odds_away):
        home_prob = straight_line(odds_home, *popt_home)
        away_prob = straight_line(odds_away, *popt_away)
        draw_prob = straight_line(odds_draw, *popt_draw)
        goal_diff = straight_line(home_prob - away_prob, *pop_diff)
        if (away_prob > home_prob) and (away_prob > draw_prob):
            away_goals = int(1 - np.round(goal_diff))
            away_goals = away_goals if away_goals >= 0 else 0
            return [str(1) + '-' + str(away_goals), away_prob]
        home_goals = int(1 + np.round(goal_diff))
        return [str(home_goals) + '-' + str(1), home_prob]

    return pick_by_odds_prob


# model from https://github.com/akrooss/KickTippTipper/blob/master/tipper.py
def calc_results_kicktipptipper(odds_home, odds_draw, odds_away):
    """By considering odds, calculates match results"""

    # Possible Results, modify at will
    deuce = [1, 1]
    team1_win = [2, 1]
    team2_win = [1, 2]
    team1_greatwin = [3, 1]
    team2_greatwin = [1, 3]

    diff = math.fabs(odds_home - odds_away)
    if diff < 1.0:
        return deuce
    elif diff > 8.0:
        if odds_home > odds_draw:
            return team2_greatwin
        else:
            return team1_greatwin
    else:
        if odds_home > odds_draw:
            return team2_win
        else:
            return team1_win


# model from https://github.com/fpoppinga/kicktipp-bot/blob/master/src/predictor/predictor.ts
def calc_predictor_kicktipp_bot(odds_home, odds_away):
    MAX_GOALS = 6
    DOMINATION_THRESHOLD = 10
    DRAW_THRESHOLD = 0.5
    NONLINEARITY = 0.4

    difference = abs(odds_away - odds_home)

    if difference < DRAW_THRESHOLD:
        return [1, 1]

    totalGoals = min((difference / DOMINATION_THRESHOLD), 1) * MAX_GOALS
    ratio = ((odds_home / odds_away if odds_home > odds_away
              else odds_away / odds_home) / (odds_home + odds_away)) ** NONLINEARITY

    winner = round(totalGoals * ratio)
    looser = round(totalGoals * (1 - ratio))

    if winner <= looser:
        winner = winner + 1

    if odds_home > odds_away:
        return [looser, winner]
    else:
        return [winner, looser]
