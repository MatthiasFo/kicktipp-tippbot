import numpy as np
import pandas as pd
import scipy
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB

from models.model_dixon_cole import train_dixon_cole_model, dixon_coles_simulate_match, DIXON_COLE_RESULTS
from models.models_based_on_betting_odds import create_odds_probability_model
from utils.generate_and_convert_picks import convert_pick_array_to_string, convert_pick_string_to_array


def remove_smallest_goal_diffs(df_games, chunk_size=200):
    # for tipping it is better to remove some of the close games. This could be that these games can go either way.
    # So it is better to use the tendency of the "safer" picks by excluding some of the "close calls".
    if chunk_size > (df_games.shape[0] / 6):
        chunk_size = int(df_games.shape[0] / 6)
    df_sorted = df_games.sort_values(by='odds_diff')

    num_games = df_sorted.shape[0]
    num_chunks = int(num_games / chunk_size)
    chunks = np.repeat(np.linspace(0, num_chunks - 1, num_chunks), chunk_size)
    df_sorted.loc[df_sorted.index[0:len(chunks)], 'chunk'] = chunks
    # check how many entries remain and put them in another chunk or the last one
    if sum(df_sorted['chunk'].isna()) > chunk_size / 3:
        df_sorted['chunk'].fillna(num_chunks, inplace=True)
    else:
        df_sorted['chunk'].fillna(num_chunks - 1, inplace=True)

    df_result = None
    for chunk in df_sorted['chunk'].unique():
        df_chunk = df_sorted.loc[df_sorted['chunk'] == chunk, :].copy(deep=True)
        df_chunk['abs_gd'] = df_chunk['goal_diff'].abs()
        df_removed = df_chunk.sort_values(by='abs_gd', ascending=False)[:int(chunk_size * 0.90)]
        if df_result is None:
            df_result = df_removed
        else:
            df_result = df_result.append(df_removed)
    return df_result.sort_index()


def create_combined_model(df_train_538, league_to_predict):
    # train dixon cole model on this and last season results
    current_season = int(df_train_538.season.max()[:4])
    previous_season = str(current_season - 1) + '/' + str(current_season)
    df_train_games_dc = df_train_538.loc[(df_train_538.season >= previous_season) &
                                         (df_train_538.league == league_to_predict), :].copy(deep=True)
    model_params = train_dixon_cole_model(df_train_games_dc)

    # fit model based on probabilities estimated with odds
    df_train_games_odds = df_train_538.copy(deep=True)
    df_train_games_odds = remove_smallest_goal_diffs(df_train_games_odds)
    odds_prob_model = create_odds_probability_model(df_train_games_odds)

    # train goal difference xgboost model
    df_train_games_odds['spi_diff'] = df_train_games_odds['spi1'] - df_train_games_odds['spi2']
    df_train_games_odds['goal_diff'] = df_train_games_odds['FTHG'] - df_train_games_odds['FTAG']

    gd_features = ['odds_diff', 'spi_diff']
    gd_label = 'goal_diff'

    relevant_labels = (df_train_games_odds[gd_label].value_counts() > (0.01 * df_train_games_odds.shape[0]))
    relevant_labels = relevant_labels.loc[relevant_labels == True].index
    df_train_games_odds = df_train_games_odds.loc[[x in relevant_labels for x in df_train_games_odds[gd_label]], :]

    y_gd_train = df_train_games_odds[gd_label].values.astype(int)

    gd_scaler = PCA(n_components=len(gd_features))

    gd_scaler.fit(df_train_games_odds[gd_features].values)
    gd_train_scaled = gd_scaler.transform(df_train_games_odds[gd_features].values)

    classes = np.unique(y_gd_train)
    class_dict = {classes[idx]: idx for idx in range(len(classes))}

    calibrated_clf = CalibratedClassifierCV(base_estimator=GaussianNB(), cv=5)
    calibrated_clf.fit(gd_train_scaled, [class_dict[x] for x in y_gd_train])

    prob_results = [convert_pick_array_to_string(x) for x in DIXON_COLE_RESULTS]

    def simulate_game(home_team, away_team, iwh, iwa, spi1, spi2, proj1, proj2):
        X = np.array([iwh - iwa, spi1 - spi2]).reshape(1, -1)
        x_pred_scaled = gd_scaler.transform(X)
        ypred = calibrated_clf.predict_proba(x_pred_scaled)
        df_pred_prob = pd.DataFrame(ypred, columns=classes)

        pick_odds_model = odds_prob_model(iwh, iwa)

        prob_spi = []
        prob_odds_model = []
        prob_classifier = []
        for result in prob_results:
            result_array = convert_pick_string_to_array(result)

            proj_home_prob = scipy.stats.poisson.pmf(result_array[0], proj1)
            proj_away_prob = scipy.stats.poisson.pmf(result_array[1], proj2)

            log_fit_home_prob = scipy.stats.poisson.pmf(result_array[0], pick_odds_model[0])
            log_fit_away_prob = scipy.stats.poisson.pmf(result_array[1], pick_odds_model[1])

            goal_diff = result_array[0] - result_array[1]
            key_list = [[abs(goal_diff - y), y] for y in df_pred_prob.columns]
            idx_key = np.argmin([y[0] for y in key_list])
            goal_diff_prob = df_pred_prob[key_list[idx_key][1]].values[0]

            prob_classifier.append(goal_diff_prob)
            prob_spi.append(proj_home_prob * proj_away_prob)
            prob_odds_model.append(log_fit_away_prob * log_fit_home_prob)

        prob_spi_norm = prob_spi / sum(prob_spi)
        prob_log_fit_norm = prob_odds_model / sum(prob_odds_model)
        prob_classifier_norm = prob_classifier / sum(prob_classifier)

        # dixon cole model prediction
        probs, lambda_k, mu_k = dixon_coles_simulate_match(model_params, home_team, away_team)
        if (lambda_k > 0.1) and (mu_k > 0.1) and len(probs) == DIXON_COLE_RESULTS.shape[0]:
            probs_dc_norm = probs / sum(probs)
            prob_combined = probs_dc_norm * prob_classifier_norm + \
                            prob_spi_norm * prob_classifier_norm + prob_log_fit_norm * prob_classifier_norm
        else:
            print('The Dixon Cole prediction went wrong!')
            prob_combined = prob_spi_norm * prob_classifier_norm + prob_log_fit_norm * prob_classifier_norm
        prob_combined_norm = prob_combined / sum(prob_combined)
        return prob_combined_norm

    def pick_games_max_prob(home_team, away_team, iwh, iwd, iwa, spi1, spi2, proj1, proj2):
        sim_result = simulate_game(home_team, away_team, iwh, iwa, spi1, spi2, proj1, proj2)
        # give the result of the spi model a bonus to honor that it is the most sophisticated model.
        # The bonus was tuned for the best kicktipp result.
        result_list_adj_proj1 = [sim_result[idx] * 1.05 if DIXON_COLE_RESULTS[idx][0] == np.round(proj1)
                                 else sim_result[idx] for idx in range(len(sim_result))]
        result_list_adj_proj2 = [result_list_adj_proj1[idx] * 1.05
                                 if (DIXON_COLE_RESULTS[idx][1] == np.round(proj2))
                                 else result_list_adj_proj1[idx] for idx in range(len(result_list_adj_proj1))]
        result_list = [[a, b] for a, b in zip(DIXON_COLE_RESULTS, result_list_adj_proj2)]
        return sorted(result_list, key=lambda l: l[1], reverse=True)

    return pick_games_max_prob
