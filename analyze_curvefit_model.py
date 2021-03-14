import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from models.basic_math_functions import logistic_fun, non_neg_logistic
from load_data import load_league_data_with_odds

df_seasons = load_league_data_with_odds(range(2006, 2021), ['D1', 'E0'])

#####################################################
# Plot the curve fit results for the logistic model #
#####################################################
df_seasons['rounded_odds_diff'] = df_seasons['odds_diff'].round().astype('int')
df_plot_odds = df_seasons.loc[:, ['rounded_odds_diff', 'FTHG', 'goal_diff']] \
    .groupby(by='rounded_odds_diff') \
    .agg(mf_hg=('FTHG', lambda x: x.value_counts().index[0]),
         mean_hg=('FTHG', 'mean'),
         mf_ag=('goal_diff', lambda x: x.value_counts().index[0]),
         mean_ag=('goal_diff', 'mean'),
         count=('FTHG', 'count'))
df_plot_odds_reduced = df_plot_odds.loc[df_plot_odds.loc[:, 'count'] > 10]

popt_log_home, pcov_log_home = curve_fit(non_neg_logistic, df_seasons['odds_diff'], df_seasons['FTHG'])
popt_log_goaldiff, pcov_log_goaldiff = curve_fit(logistic_fun, df_seasons['odds_diff'], df_seasons['goal_diff'])

plt.plot(df_plot_odds.index, np.round(non_neg_logistic(df_plot_odds.index, *popt_log_home)), 'b', label='home', linewidth=2)
plt.plot(df_plot_odds.index, np.round(logistic_fun(df_plot_odds.index, *popt_log_goaldiff)), 'r--', label='away', linewidth=2)
plt.ylabel('Predicted goals')
plt.xlabel('Odds difference (Home odds - away odds)')
plt.legend()
plt.show()

plt.subplot(211)
plt.title(r'Goals over odds differential')
plt.scatter(x=df_seasons['odds_diff'], y=df_seasons['FTHG'], color='blue')
plt.plot(df_plot_odds_reduced.index, df_plot_odds_reduced.loc[:, 'mf_hg'], 'm-', label='most frequent', linewidth=3)
plt.plot(df_plot_odds_reduced.index, df_plot_odds_reduced.loc[:, 'mean_hg'], 'r-', label='mean', linewidth=2)
plt.plot(df_plot_odds.index, non_neg_logistic(df_plot_odds.index, *popt_log_home), 'g--', label='fit', linewidth=2)
plt.ylabel('Home goals')
plt.xlabel('Odds difference (Home odds - away odds)')
plt.legend()

plt.subplot(212)
plt.scatter(x=df_seasons['odds_diff'], y=df_seasons['goal_diff'], color='blue')
plt.plot(df_plot_odds_reduced.index, df_plot_odds_reduced.loc[:, 'mf_ag'], 'm-', label='most frequent', linewidth=3)
plt.plot(df_plot_odds_reduced.index, df_plot_odds_reduced.loc[:, 'mean_ag'], 'r-', label='mean', linewidth=2)
plt.plot(df_plot_odds.index, logistic_fun(df_plot_odds.index, *popt_log_goaldiff), 'g--', label='fit', linewidth=3)
plt.ylabel('Away goals')
plt.xlabel('Odds difference (Home odds - away odds)')
plt.legend()
plt.show()

#################################################
# Check if individual teams differ from the fit #
#################################################
df_at_bayern = df_seasons.loc[df_seasons['HomeTeam'] == 'Bayern Munich', :].copy(deep=True)
df_at_bayern['rounded_odds_diff'] = df_at_bayern['odds_diff'].round().astype('int')
df_plot_at_bayern = df_at_bayern.loc[:, ['rounded_odds_diff', 'FTHG', 'FTAG']] \
    .groupby(by='rounded_odds_diff') \
    .agg(mf_hg=('FTHG', lambda x: x.value_counts().index[0]),
         mean_hg=('FTHG', 'mean'),
         mf_ag=('FTAG', lambda x: x.value_counts().index[0]),
         mean_ag=('FTAG', 'mean'),
         count=('FTHG', 'count'))
df_plot_at_bayern_red = df_plot_at_bayern.loc[df_plot_at_bayern.loc[:, 'count'] > 10]

plt.subplot(221)
plt.title(r'Goals over odds differential')
plt.scatter(x=df_at_bayern['odds_diff'], y=df_at_bayern['FTHG'], color='blue')
plt.plot(df_plot_at_bayern_red.index, df_plot_at_bayern_red.loc[:, 'mean_hg'], 'r-', label='mean at bayern', linewidth=2)
plt.plot(df_plot_odds_reduced.index, df_plot_odds_reduced.loc[:, 'mean_hg'], 'g--', label='mean overall', linewidth=2)
plt.ylabel('Home goals')
plt.xlabel('Odds difference (Home odds - away odds)')
plt.legend()

plt.subplot(222)
plt.scatter(x=df_at_bayern['odds_diff'], y=df_at_bayern['FTAG'], color='blue')
plt.plot(df_plot_at_bayern_red.index, df_plot_at_bayern_red.loc[:, 'mean_ag'], 'r-', label='mean at bayern', linewidth=2)
plt.plot(df_plot_odds_reduced.index, df_plot_odds_reduced.loc[:, 'mean_ag'], 'g--', label='mean overall', linewidth=2)
plt.ylabel('Away goals')
plt.xlabel('Odds difference (Home odds - away odds)')
plt.legend()

df_road_bayern = df_seasons.loc[df_seasons['AwayTeam'] == 'Bayern Munich', :].copy(deep=True)
df_road_bayern['rounded_odds_diff'] = df_road_bayern['odds_diff'].round().astype('int')
df_plot_road_bayern = df_road_bayern.loc[:, ['rounded_odds_diff', 'FTHG', 'FTAG']] \
    .groupby(by='rounded_odds_diff') \
    .agg(mf_hg=('FTHG', lambda x: x.value_counts().index[0]),
         mean_hg=('FTHG', 'mean'),
         mf_ag=('FTAG', lambda x: x.value_counts().index[0]),
         mean_ag=('FTAG', 'mean'),
         count=('FTHG', 'count'))
df_plot_road_bayern_red = df_plot_road_bayern.loc[df_plot_road_bayern.loc[:, 'count'] > 10]

plt.subplot(223)
plt.title(r'Goals over odds differential')
plt.scatter(x=df_road_bayern['odds_diff'], y=df_road_bayern['FTHG'], color='blue')
plt.plot(df_plot_road_bayern_red.index, df_plot_road_bayern_red.loc[:, 'mean_hg'], 'r-', label='mean road bayern', linewidth=2)
plt.plot(df_plot_odds_reduced.index, df_plot_odds_reduced.loc[:, 'mean_hg'], 'g--', label='mean overall', linewidth=2)
plt.ylabel('Home goals')
plt.xlabel('Odds difference (Home odds - away odds)')
plt.legend()

plt.subplot(224)
plt.scatter(x=df_road_bayern['odds_diff'], y=df_road_bayern['FTAG'], color='blue')
plt.plot(df_plot_road_bayern_red.index, df_plot_road_bayern_red.loc[:, 'mean_ag'], 'r-', label='mean road bayern', linewidth=2)
plt.plot(df_plot_odds_reduced.index, df_plot_odds_reduced.loc[:, 'mean_ag'], 'g--', label='mean overall', linewidth=2)
plt.ylabel('Away goals')
plt.xlabel('Odds difference (Home odds - away odds)')
plt.legend()

plt.show()


#######################################
# Check the fit of the logistic model #
#######################################
def plot_compare_hist(odds_differential, side='Home'):
    if side == 'Home':
        poi_lambda = np.round(non_neg_logistic(odds_differential, *popt_log_home))
        goal_column = 'FTHG'
    else:
        poi_lambda = np.round(non_neg_logistic(odds_differential, *popt_log_home) - logistic_fun(odds_differential, *popt_log_goaldiff))
        goal_column = 'FTAG'
    num_samples = df_seasons.loc[df_seasons['rounded_odds_diff'] == odds_differential, 'FTAG'].shape[0]
    plt.hist(df_seasons.loc[df_seasons['rounded_odds_diff'] == odds_differential, goal_column],
             histtype='stepfilled', bins=35, alpha=0.5,
             label=r'Sample', color='#7A68A6')
    plt.hist(np.random.poisson(poi_lambda, num_samples),
             histtype='stepfilled', bins=35, alpha=0.5,
             label=r'Poission estimate', color='#A60628')
    plt.title(side + ' | odds-diff: ' + str(odds_differential))


plt.subplot(321)
plot_compare_hist(-5.0, 'Home')
plt.subplot(323)
plot_compare_hist(0.0, 'Home')
plt.subplot(325)
plot_compare_hist(5.0, 'Home')
plt.subplot(322)
plot_compare_hist(-5.0, 'Away')
plt.legend()
plt.subplot(324)
plot_compare_hist(0.0, 'Away')
plt.subplot(326)
plot_compare_hist(5.0, 'Away')
plt.show()


############################################################
# Check the additional predictive quality of the draw odds #
############################################################
def plot_agg(df_data, side='home'):
    df_plot = df_data.copy(deep=True)
    if side == 'home':
        pcolor = 'blue'
        mean_line = 'g-'
    else:
        pcolor = 'red'
        mean_line = 'g--'
    df_plot['rounded_IWD'] = df_plot['IWD'].round(decimals=1)
    df_plot_draw = df_plot.loc[:, ['rounded_IWD', 'goal_diff', 'odds_diff']] \
        .groupby(by='rounded_IWD') \
        .agg(mean_gd=('goal_diff', 'mean'),
             mean_odds_diff=('odds_diff', 'mean'),
             count=('goal_diff', 'count'))
    df_plot_draw_reduced = df_plot_draw.loc[df_plot_draw.loc[:, 'count'] > 10]
    plt.scatter(x=df_plot['IWD'], y=df_plot['goal_diff'], color=pcolor, alpha=0.2)
    plt.plot(df_plot_draw_reduced.index, df_plot_draw_reduced.loc[:, 'mean_gd'], mean_line, label='mean ' + side,
             linewidth=3)


home_favorite = df_seasons.loc[df_seasons['IWH'] < df_seasons['IWA']]
away_favorite = df_seasons.loc[df_seasons['IWH'] > df_seasons['IWA']]
plt.subplot(211)
plt.scatter(x=home_favorite['IWD'], y=home_favorite['odds_diff'], label='home favorite', color='blue', alpha=0.2)
plt.scatter(x=-away_favorite['IWD'], y=away_favorite['odds_diff'], label='away favorite', color='red', alpha=0.2)
plt.ylabel('Odds difference')
plt.xlabel('Draw odds (negative for away favorite)')
plt.legend()
plt.subplot(212)
plt.title(r'Goal difference over odds for draw')
plot_agg(home_favorite, 'home')
plot_agg(away_favorite, 'away')
plt.ylabel('Goal difference')
plt.xlabel('Draw odds')
plt.legend()
plt.show()