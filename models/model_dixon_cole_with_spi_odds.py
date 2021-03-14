import numpy as np
import pandas as pd
import scipy
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB

from models.model_dixon_cole import train_dixon_cole_model, dixon_coles_simulate_match, DIXON_COLE_RESULTS
from models.models_based_on_betting_odds import create_curvefit_model
from utils.generate_and_convert_picks import convert_pick_array_to_string, convert_pick_string_to_array


def create_combined_model(df_train_538, league_to_predict):
    # train dixon cole model on this and last season results
    current_season = int(df_train_538.season.max()[:4])
    previous_season = str(current_season - 1) + '/' + str(current_season)
    df_train_games_dc = df_train_538.loc[(df_train_538.season >= previous_season) &
                                         (df_train_538.league >= league_to_predict), :].copy(deep=True)
    dc_trained_teams = set(np.concatenate((df_train_games_dc['team1'].values,
                                           df_train_games_dc['team2'].values)))
    model_params = train_dixon_cole_model(df_train_games_dc)

    # fit logistic curves
    log_fit_model = create_curvefit_model(df_train_538)

    # train goal difference xgboost model
    df_train_538['spi_diff'] = df_train_538['spi1'] - df_train_538['spi2']
    df_train_538['goal_diff'] = df_train_538['FTHG'] - df_train_538['FTAG']

    gd_features = ['odds_diff', 'spi_diff']
    gd_label = 'goal_diff'
    df_train = df_train_538.copy(deep=True)

    relevant_labels = (df_train[gd_label].value_counts() > (0.01 * df_train.shape[0]))
    relevant_labels = relevant_labels.loc[relevant_labels == True].index
    df_train = df_train.loc[[x in relevant_labels for x in df_train[gd_label]], :]

    y_gd_train = df_train[gd_label].values.astype(int)

    gd_scaler = PCA(n_components=len(gd_features))

    gd_scaler.fit(df_train[gd_features].values)
    gd_train_scaled = gd_scaler.transform(df_train[gd_features].values)

    classes = np.unique(y_gd_train)
    class_dict = {classes[idx]: idx for idx in range(len(classes))}

    calibrated_clf = CalibratedClassifierCV(base_estimator=GaussianNB(), cv=5)
    calibrated_clf.fit(gd_train_scaled, [class_dict[x] for x in y_gd_train])

    prob_results = [convert_pick_array_to_string(x) for x in DIXON_COLE_RESULTS]

    def simulate_game(home_team, away_team, iwh, iwa, spi1, spi2, proj1, proj2):
        # classifier prediction for goal difference
        X = np.array([iwh - iwa, spi1 - spi2]).reshape(1, -1)
        x_pred_scaled = gd_scaler.transform(X)
        ypred = calibrated_clf.predict_proba(x_pred_scaled)
        df_pred_prob = pd.DataFrame(ypred, columns=classes)

        pick_log_fit = log_fit_model(iwh - iwa)

        prob_spi = []
        prob_log_fit = []
        prob_classifier = []
        for result in prob_results:
            result_array = convert_pick_string_to_array(result)

            proj_home_prob = scipy.stats.poisson.pmf(result_array[0], proj1)
            proj_away_prob = scipy.stats.poisson.pmf(result_array[1], proj2)

            log_fit_home_prob = scipy.stats.poisson.pmf(result_array[0], pick_log_fit[0])
            log_fit_away_prob = scipy.stats.poisson.pmf(result_array[1], pick_log_fit[1])

            goal_diff = result_array[0] - result_array[1]
            key_list = [[abs(goal_diff - y), y] for y in df_pred_prob.columns]
            idx_key = np.argmin([y[0] for y in key_list])
            goal_diff_prob = df_pred_prob[key_list[idx_key][1]].values[0]

            prob_classifier.append(goal_diff_prob)
            prob_spi.append(proj_home_prob * proj_away_prob)
            prob_log_fit.append(log_fit_away_prob * log_fit_home_prob)

        prob_spi_norm = prob_spi / sum(prob_spi)
        prob_log_fit_norm = prob_log_fit / sum(prob_log_fit)
        prob_classifier_norm = prob_classifier / sum(prob_classifier)
        prob_spi_odds_model = prob_spi_norm * prob_classifier_norm + prob_log_fit_norm * prob_classifier_norm
        prob_spi_odds_model_norm = prob_spi_odds_model / sum(prob_spi_odds_model)

        # dixon cole model prediction
        probs_dc_norm = None
        if (home_team in dc_trained_teams) and (away_team in dc_trained_teams):
            probs, lambda_k, mu_k = dixon_coles_simulate_match(model_params, home_team, away_team)
            if (lambda_k > 0.1) and (mu_k > 0.1) and len(probs) == DIXON_COLE_RESULTS.shape[0]:
                probs_dc_norm = probs / sum(probs)
            else:
                print('The Dixon Cole prediction went wrong!')
        if probs_dc_norm is None:
            prob_combined_norm = prob_spi_odds_model_norm
        else:
            prob_combined = probs_dc_norm * prob_classifier_norm + \
                            prob_spi_norm * prob_classifier_norm + prob_log_fit_norm * prob_classifier_norm
            prob_combined_norm = prob_combined / sum(prob_combined)
        return [prob_combined_norm, probs_dc_norm, prob_spi_norm, prob_log_fit_norm, prob_classifier_norm]

    def pick_games_max_prob(home_team, away_team, iwh, iwd, iwa, spi1, spi2, proj1, proj2):
        sim_result = simulate_game(home_team, away_team, iwh, iwa, spi1, spi2, proj1, proj2)
        # reduce value of draws so they are picked only if the simulation is really confident in the draw
        # since you only get points if the game actually has a goal difference of zero.
        # There are more opportunities for a pick on a win can generate points.
        result_list_red_draw = [sim_result[0][idx] * 0.68 if DIXON_COLE_RESULTS[idx][0] == DIXON_COLE_RESULTS[idx][1]
                                else sim_result[0][idx] for idx in range(len(sim_result[0]))]
        result_list_adj_proj1 = [result_list_red_draw[idx] * 1.08 if DIXON_COLE_RESULTS[idx][0] == np.round(proj1)
                                else result_list_red_draw[idx] for idx in range(len(result_list_red_draw))]
        result_list_adj_proj2 = [result_list_adj_proj1[idx] * 1.08
                                 if (DIXON_COLE_RESULTS[idx][1] == np.round(proj2))
                                 else result_list_adj_proj1[idx] for idx in range(len(result_list_adj_proj1))]
        result_list = [[a, b] for a, b in zip(DIXON_COLE_RESULTS, result_list_adj_proj2)]
        return sorted(result_list, key=lambda l: l[1], reverse=True)
    return pick_games_max_prob
