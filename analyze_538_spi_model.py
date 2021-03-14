import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.evaluate_kicktipp_points import evaluate_kicktipp_432
from utils.slize_games_into_chunks import chunk_games
from models.basic_math_functions import logistic_fun, non_neg_logistic
from utils.generate_and_convert_picks import convert_pick_array_to_string
from load_data import load_odds_and_538_data
from utils.module_timeseries_handling import smooth_stats_by_team, shift_stats_by_team

pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)

np.random.seed(1994)

df_538_eval = load_odds_and_538_data()

df_538_eval['spi_diff'] = df_538_eval['spi1'] - df_538_eval['spi2']
df_538_eval['proj_goal_diff'] = df_538_eval['proj_score1'] - df_538_eval['proj_score2']
df_538_eval['prob_diff'] = df_538_eval['prob1'] - 1 / df_538_eval['IWH']
df_538_eval['home_error'] = df_538_eval['FTHG'] - df_538_eval['proj_score1']
df_538_eval['diff_error'] = df_538_eval['goal_diff'] - df_538_eval['proj_goal_diff']

##################################
# Plot accuracy of the SPI model #
##################################
df_538_eval['kt_points'] = [
    evaluate_kicktipp_432([home_goals, away_goals], np.round([proj1, proj2]))
    for (home_goals, away_goals, proj1, proj2) in
    zip(df_538_eval['FTHG'], df_538_eval['FTAG'], df_538_eval['proj_score1'], df_538_eval['proj_score2'])
]

df_chunked = chunk_games(df_538_eval, 'spi_diff')
df_grouped = df_chunked.groupby(by='chunk').mean()
plt.scatter(x=df_538_eval['spi_diff'], y=df_538_eval['kt_points'], color='blue')
plt.plot(df_grouped['spi_diff'], df_grouped['kt_points'], 'r')
plt.ylabel('kt-points')
plt.xlabel('spi difference')
plt.show()

#######################################################
# Plot distribution of real goals and predicted goals #
#######################################################

df_test_poisson = df_538_eval.copy(deep=True).append(df_538_eval.copy(deep=True)).append(
    df_538_eval.copy(deep=True)).append(df_538_eval.copy(deep=True))

df_test_poisson['proj_test1'] = [np.random.poisson(pred, 1)[0] for pred in df_test_poisson['proj_score1']]
df_test_poisson['proj_test2'] = [np.random.poisson(pred, 1)[0] for pred in df_test_poisson['proj_score2']]

df_hist = pd.DataFrame(df_test_poisson['FTHG'].round().value_counts()) \
    .join(df_test_poisson['proj_test1'].round().value_counts())
ax1 = plt.subplot(121)
df_hist.sort_index().plot.bar(ax=ax1)
plt.legend()
df_hist = pd.DataFrame(df_test_poisson['FTAG'].round().value_counts()) \
    .join(df_test_poisson['proj_test2'].round().value_counts())
ax2 = plt.subplot(122)
df_hist.sort_index().plot.bar(ax=ax2)
plt.legend()
plt.show()

df_test_poisson['result_spi'] = [convert_pick_array_to_string([x, y]) for x, y in
                                 zip(df_test_poisson['proj_test1'].round(), df_test_poisson['proj_test2'].round())]
df_hist = pd.DataFrame(df_test_poisson['result'].value_counts()) \
    .join(df_test_poisson['result_spi'].value_counts())
ax = plt.figure()
df_hist.sort_index().plot.bar(figsize=(18, 6))
plt.show()

####################################################################
# Check if there is any correlation with winning or losing streaks #
####################################################################
df_538_eval['win_home'] = (df_538_eval['FTHG'] > df_538_eval['FTAG']).astype('float')
df_538_eval['win_away'] = (df_538_eval['FTHG'] < df_538_eval['FTAG']).astype('float')

df_streaks = smooth_stats_by_team(df_538_eval.copy(deep=True), 'win', win_length=4)
df_streaks = shift_stats_by_team(df_streaks, 'win')

df_streaks['proj_error1'] = df_streaks['FTHG'] - df_streaks['proj_score1']
df_streaks['proj_error2'] = df_streaks['FTAG'] - df_streaks['proj_score2']

plt.subplot(211)
df_grouped = df_streaks.groupby(by='win_home').mean()
plt.scatter(x=df_streaks['win_home'], y=df_streaks['proj_error1'], color='blue')
plt.plot(df_grouped.index, df_grouped['proj_error1'], 'r')
plt.plot([0, 1], [0, 0], 'k-', linewidth=2)
plt.xlabel('home streak')
plt.ylabel('proj1 error')
plt.subplot(212)
df_grouped = df_streaks.groupby(by='win_away').mean()
plt.scatter(x=df_streaks['win_away'], y=df_streaks['proj_error1'], color='blue')
plt.plot(df_grouped.index, df_grouped['proj_error1'], 'r')
plt.plot([0, 1], [0, 0], 'k-', linewidth=2)
plt.xlabel('away streak')
plt.ylabel('proj1 error')

###################################################
# Check possible correlations of the models error #
###################################################
print(df_538_eval.loc[:, ['diff_error', 'prob_diff', 'proj_goal_diff', 'odds_diff', 'IWD']].corr()['diff_error'])
mut_info = mutual_info_regression(df_538_eval[['prob_diff', 'proj_goal_diff', 'odds_diff', 'IWD']].values,
                                  df_538_eval['diff_error'].values)

plt.subplot(221)
plt.scatter(x=df_538_eval['prob_diff'], y=df_538_eval['diff_error'], color='blue')
df_plot_odds = chunk_games(df_538_eval, chunk_by='prob_diff').groupby(by='chunk').mean()
plt.plot(df_plot_odds['prob_diff'], df_plot_odds['diff_error'], 'r-')
plt.ylabel('goal diff error')
plt.xlabel('prob_diff')
plt.subplot(222)
plt.scatter(x=df_538_eval['proj_goal_diff'], y=df_538_eval['diff_error'], color='blue')
df_plot_odds = chunk_games(df_538_eval, chunk_by='proj_goal_diff').groupby(by='chunk').mean()
plt.plot(df_plot_odds['proj_goal_diff'], df_plot_odds['diff_error'], 'r-')
plt.ylabel('goal diff error')
plt.xlabel('proj_goal_diff')
plt.subplot(223)
plt.scatter(x=df_538_eval['odds_diff'], y=df_538_eval['diff_error'], color='blue')
df_plot_odds = chunk_games(df_538_eval, chunk_by='odds_diff').groupby(by='chunk').mean()
plt.plot(df_plot_odds['odds_diff'], df_plot_odds['diff_error'], 'r-')
plt.ylabel('goal diff error')
plt.xlabel('odds_diff')
plt.subplot(224)
plt.scatter(x=df_538_eval['IWD'], y=df_538_eval['diff_error'], color='blue')
df_plot_odds = chunk_games(df_538_eval, chunk_by='IWD').groupby(by='chunk').mean()
plt.plot(df_plot_odds['IWD'], df_plot_odds['diff_error'], 'r-')
plt.ylabel('goal diff error')
plt.xlabel('IWD')
plt.show()

# data coordinates and values
x_feature = 'odds_diff'
y_feature = 'proj_goal_diff'
x = df_538_eval[x_feature].values.reshape(-1, 1)
y = df_538_eval[y_feature].values.reshape(-1, 1)
z = df_538_eval['diff_error'].values

scaler_x = StandardScaler()
scaler_x.fit(x)
x_scaled = scaler_x.transform(x)[:, 0]

scaler_y = StandardScaler()
scaler_y.fit(y)
y_scaled = scaler_y.transform(y)[:, 0]

# target grid to interpolate to
xi = np.arange(min(x_scaled), max(x_scaled), 0.1)
yi = np.arange(min(y_scaled), max(y_scaled), 0.1)
xi, yi = np.meshgrid(xi, yi)

# interpolate
zi = griddata((x_scaled, y_scaled), z, (xi, yi), method='linear')

# plot
fig = plt.figure()
ax = fig.add_subplot(111)
plt.contourf(xi, yi, zi, np.arange(-3, 3, 0.1))
# plt.plot(x, y, 'k.')
plt.xlabel(x_feature, fontsize=16)
plt.ylabel(y_feature, fontsize=16)
plt.title('SPI Error in goal difference')
cbar = plt.colorbar()
plt.show()

########################################################
# Evaluate the models with a test and training dataset #
########################################################
df_training_base = df_538_eval.loc[
                   (df_538_eval['season'] < '2019/2020') | (df_538_eval['league'] != 'D1'), :].copy(deep=True)
df_test_d1 = df_538_eval.loc[
             (df_538_eval['season'] >= '2019/2020') & (df_538_eval['league'] == 'D1'), :].copy(deep=True)
df_training_base.set_index(['league', 'season', 'game_day'], inplace=True, drop=False)
train_split = train_test_split([str(x) for x in df_training_base.index.unique().values], test_size=0.20,
                               random_state=1)
df_test = df_training_base.loc[[str(x) in train_split[1] for x in df_training_base.index], :].append(
    df_test_d1).reset_index(drop=True).sort_values('datetime')
df_test = df_test.sort_values('datetime')
df_training = df_538_eval.loc[[str(x) not in train_split[1] for x in df_538_eval.index], :].reset_index(drop=True)

###########################################################################
# Plot the curve fit results for the logistic model with different inputs #
###########################################################################
df_chunked_538 = chunk_games(df_538_eval, 'proj_goal_diff')
df_plot_odds = df_chunked_538.loc[:, ['chunk', 'proj_goal_diff', 'FTHG', 'goal_diff']] \
    .groupby(by='chunk') \
    .agg(mf_hg=('FTHG', lambda x: x.value_counts().index[0]),
         mean_hg=('FTHG', 'mean'),
         mf_gd=('goal_diff', lambda x: x.value_counts().index[0]),
         mean_gd=('goal_diff', 'mean'),
         mean_proj_gd=('proj_goal_diff', 'mean'),
         count=('FTHG', 'count'))

popt_log_home, pcov_log_home = curve_fit(non_neg_logistic, df_538_eval['proj_goal_diff'], df_538_eval['FTHG'])
popt_log_goaldiff, pcov_log_goaldiff = curve_fit(logistic_fun, df_538_eval['proj_goal_diff'], df_538_eval['goal_diff'])

plt.figure(figsize=(12, 8))
error_of_fit = mean_squared_error(df_538_eval['FTHG'], non_neg_logistic(df_538_eval['proj_goal_diff'], *popt_log_home))
plt.subplot(221)
plt.scatter(x=df_538_eval['proj_goal_diff'], y=df_538_eval['FTHG'], color='blue')
plt.plot(df_plot_odds['mean_proj_gd'], df_plot_odds['mf_hg'], 'm-', label='most frequent', linewidth=3)
plt.plot(df_plot_odds['mean_proj_gd'], df_plot_odds['mean_hg'], 'r-', label='mean', linewidth=2)
plt.plot(df_plot_odds['mean_proj_gd'], non_neg_logistic(df_plot_odds['mean_proj_gd'], *popt_log_home), 'g--',
         label='fit',
         linewidth=2)
plt.ylabel('Home goals')
plt.xlabel('Proj. goal difference')
plt.title('RMSE: ' + str(error_of_fit))
plt.legend()

error_of_fit = mean_squared_error(df_538_eval['goal_diff'],
                                  logistic_fun(df_538_eval['proj_goal_diff'], *popt_log_goaldiff))
plt.subplot(223)
plt.scatter(x=df_538_eval['proj_goal_diff'], y=df_538_eval['goal_diff'], color='blue')
plt.plot(df_plot_odds['mean_proj_gd'], df_plot_odds['mf_gd'], 'm-', label='most frequent', linewidth=3)
plt.plot(df_plot_odds['mean_proj_gd'], df_plot_odds['mean_gd'], 'r-', label='mean', linewidth=2)
plt.plot(df_plot_odds['mean_proj_gd'], logistic_fun(df_plot_odds['mean_proj_gd'], *popt_log_goaldiff), 'g--',
         label='fit',
         linewidth=3)
plt.ylabel('Goal difference')
plt.xlabel('Proj. goal difference')
plt.title('RMSE: ' + str(error_of_fit))
plt.legend()

df_chunked_538 = chunk_games(df_538_eval, 'odds_diff')
df_plot_odds = df_chunked_538.loc[:, ['chunk', 'odds_diff', 'FTHG', 'goal_diff']] \
    .groupby(by='chunk') \
    .agg(mf_hg=('FTHG', lambda x: x.value_counts().index[0]),
         mean_hg=('FTHG', 'mean'),
         mf_gd=('goal_diff', lambda x: x.value_counts().index[0]),
         mean_gd=('goal_diff', 'mean'),
         mean_odds=('odds_diff', 'mean'),
         count=('FTHG', 'count'))

popt_log_home, pcov_log_home = curve_fit(non_neg_logistic, df_538_eval['odds_diff'], df_538_eval['FTHG'])
popt_log_goaldiff, pcov_log_goaldiff = curve_fit(logistic_fun, df_538_eval['odds_diff'], df_538_eval['goal_diff'])

error_of_fit = mean_squared_error(df_538_eval['FTHG'], non_neg_logistic(df_538_eval['odds_diff'], *popt_log_home))
plt.subplot(222)
plt.scatter(x=df_538_eval['odds_diff'], y=df_538_eval['FTHG'], color='blue')
plt.plot(df_plot_odds['mean_odds'], df_plot_odds['mf_hg'], 'm-', label='most frequent', linewidth=3)
plt.plot(df_plot_odds['mean_odds'], df_plot_odds['mean_hg'], 'r-', label='mean', linewidth=2)
plt.plot(df_plot_odds['mean_odds'], non_neg_logistic(df_plot_odds['mean_odds'], *popt_log_home), 'g--', label='fit',
         linewidth=2)
plt.ylabel('Home goals')
plt.xlabel('Odds difference')
plt.title('RMSE: ' + str(error_of_fit))
plt.legend()

error_of_fit = mean_squared_error(df_538_eval['goal_diff'], logistic_fun(df_538_eval['odds_diff'], *popt_log_goaldiff))
plt.subplot(224)
plt.scatter(x=df_538_eval['odds_diff'], y=df_538_eval['goal_diff'], color='blue')
plt.plot(df_plot_odds['mean_odds'], df_plot_odds['mf_gd'], 'm-', label='most frequent', linewidth=3)
plt.plot(df_plot_odds['mean_odds'], df_plot_odds['mean_gd'], 'r-', label='mean', linewidth=2)
plt.plot(df_plot_odds['mean_odds'], logistic_fun(df_plot_odds['mean_odds'], *popt_log_goaldiff), 'g--', label='fit',
         linewidth=3)
plt.ylabel('Goal difference')
plt.xlabel('Odds difference')
plt.legend()
plt.title('RMSE: ' + str(error_of_fit))
plt.show()