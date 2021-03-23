# https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/
# https://github.com/dashee87/blogScripts/blob/master/Python/2018-09-13-dixon-coles-and-time-weighting/dixon_coles_decay_xi_5season.py

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds
from scipy.special import factorial
from scipy.stats import poisson

from utils.generate_and_convert_picks import get_likely_results

warnings.filterwarnings('error')
pd.set_option('display.width', None)

DIXON_COLE_RESULTS = get_likely_results()


def dixon_coles_simulate_match(params_dict, home_team, away_team):
    min_attack = min([val for key, val in params_dict.items() if 'attack_' in key])
    max_defense = min([val for key, val in params_dict.items() if 'defence_' in key])
    if (home_team in params_dict['trained_teams']):
        alpha_x = params_dict['attack_' + home_team]
        beta_x = params_dict['defence_' + home_team]
    else:
        # assume the team newly joined the league from a lower league -> take worst values
        alpha_x = min_attack
        beta_x = max_defense
    if (away_team in params_dict['trained_teams']):
        alpha_y = params_dict['attack_' + away_team]
        beta_y = params_dict['defence_' + away_team]
    else:
        # assume the team newly joined the league from a lower league -> take worst values
        alpha_y = min_attack
        beta_y = max_defense
    gamma = params_dict['home_adv']

    lambda_k = alpha_x * beta_y * gamma
    mu_k = alpha_y * beta_x

    prob_results = [poisson.pmf(result[0], lambda_k) *
                    poisson.pmf(result[1], mu_k) for result in DIXON_COLE_RESULTS]
    return prob_results, lambda_k, mu_k


def pick_with_dixon_cole(params_dict, homeTeam, awayTeam):
    result_probabilities, lambda_k, mu_k = dixon_coles_simulate_match(params_dict, homeTeam, awayTeam)
    pick_list = [[a, b] for a, b in zip(DIXON_COLE_RESULTS, result_probabilities)]
    return sorted(pick_list, key=lambda l: l[1], reverse=True), lambda_k, mu_k


def train_dixon_cole_model(df_train_dataset, init_vals=None):
    teams = np.sort(np.unique(np.concatenate(
        (df_train_dataset['team1'].values, df_train_dataset['team2'].values)
    )))
    n_teams = len(teams)

    def init_params():
        # random initialisation of model parameters
        off_strengs = np.random.uniform(0.5, 1.5, (n_teams))  # alpha (attack strength)
        dev_strengs = np.random.uniform(0.5, 1.5, (n_teams))  # beta (defence strength)
        random_vals = np.concatenate((off_strengs / np.mean(off_strengs),  # normalize to fit constraint
                                      dev_strengs / np.mean(dev_strengs),
                                      np.array([1.0])  # rho (score correction), gamma (home advantage)
                                      ))
        return random_vals

    df_train_dataset['time_diff'] = (max(df_train_dataset['datetime']) - df_train_dataset['datetime']).dt.days

    vec_home_goals = df_train_dataset.FTHG.values.astype(int)
    vec_away_goals = df_train_dataset.FTAG.values.astype(int)
    vec_home_goals_fact = factorial(vec_home_goals)
    vec_away_goals_fact = factorial(vec_away_goals)
    vec_home_teams = df_train_dataset.team1.values
    vec_away_teams = df_train_dataset.team2.values

    # a team strength should remain constant over a half of a season so weigh it equally
    current_season = df_train_dataset.season.max()
    num_game_days = df_train_dataset.game_day.max()
    season_half = int(num_game_days / 2)
    current_game_day = df_train_dataset.loc[df_train_dataset.season != current_season, 'game_day'].max()
    time_decay_prev_season = 0.66 * (df_train_dataset.season != current_season)
    if current_game_day > season_half:
        time_decay_curr_season = 0.8 * (df_train_dataset.season == current_season) * \
                                 (df_train_dataset.game_day <= season_half) + \
                                 1.0 * \
                                 (df_train_dataset.season == current_season) * \
                                 (df_train_dataset.game_day > season_half)
    else:
        time_decay_curr_season = 1.0 * (df_train_dataset.season == current_season)

    time_decay = time_decay_curr_season + time_decay_prev_season

    def estimate_paramters(params):
        gamma = params[-1:]

        score_coef_dict = dict(zip(teams, params[:n_teams]))
        defend_coef_dict = dict(zip(teams, params[n_teams:(2 * n_teams)]))

        alpha_x = np.array([score_coef_dict[x] for x in vec_home_teams])
        alpha_y = np.array([score_coef_dict[x] for x in vec_away_teams])

        beta_x = np.array([defend_coef_dict[x] for x in vec_home_teams])
        beta_y = np.array([defend_coef_dict[x] for x in vec_away_teams])

        lambda_k = alpha_x * beta_y * gamma
        mu_k = alpha_y * beta_x

        likelyhood = (np.power(lambda_k, vec_home_goals) * np.exp(-lambda_k) / vec_home_goals_fact) + \
                     (np.power(mu_k, vec_away_goals) * np.exp(-mu_k) / vec_away_goals_fact)
        log_likelyhood = np.log(likelyhood)

        objective = sum(time_decay * log_likelyhood)

        # use negative objective to minimize
        return -objective

    lower_bound = np.array([10 ** -9 for i in range(2 * n_teams)] + [10 ** -9])
    upper_bound = np.array([10 for i in range(2 * n_teams)] + [10])
    bnds = Bounds(lower_bound, upper_bound)
    if init_vals is None:
        init_vals = init_params()
    try:
        opt_output = minimize(estimate_paramters, init_vals, bounds=bnds, method='SLSQP',
                              constraints={'type': 'eq', 'fun': lambda x: sum(x[:n_teams]) - n_teams},
                              options={'disp': False, 'maxiter': 1000})
    except ValueError:
        # There was a scipy issue with x0 -> simply retry on this rare internal error
        init_vals = init_params()
        opt_output = minimize(estimate_paramters, init_vals, bounds=bnds, method='SLSQP',
                              constraints={'type': 'eq', 'fun': lambda x: sum(x[:n_teams]) - n_teams},
                              options={'disp': False, 'maxiter': 1000})
    model_params = dict(zip(['attack_' + team for team in teams] +
                            ['defence_' + team for team in teams] +
                            ['home_adv'],
                            opt_output.x))
    model_params['trained_teams'] = teams
    return model_params
