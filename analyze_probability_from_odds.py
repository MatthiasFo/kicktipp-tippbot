import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

from load_data import load_league_data_with_odds
from models.basic_math_functions import straight_line, quadratic_fun

df_data = load_league_data_with_odds(range(2006, 2021), ['I1', 'D1', 'E0', 'SP1'])

df_data['est_home_prob'] = [1 / x for x in df_data['IWH'].values]
df_data['est_away_prob'] = [1 / x for x in df_data['IWA'].values]
df_data['est_draw_prob'] = [1 / x for x in df_data['IWD'].values]

df_data['est_home_away_diff'] = df_data['est_home_prob'] - df_data['est_away_prob']

df_data['home_win'] = df_data['FTHG'] > df_data['FTAG']
df_data['away_win'] = df_data['FTHG'] < df_data['FTAG']
df_data['draw'] = df_data['FTHG'] == df_data['FTAG']

perc_limit = 99
# home win probability
ax1 = plt.subplot(221)
df_grouped = df_data[['est_home_prob', 'FTHG', 'home_win']].sort_values(by='est_home_prob').rolling(window=100, center=True).mean()
plt.scatter(df_grouped['est_home_prob'], df_grouped['home_win'], label='chunked mean', alpha=0.5)
plt.plot([0, 1], [0, 1], 'm--', label='exact match')
popt, _ = curve_fit(straight_line, df_data['est_home_prob'], df_data['home_win'], absolute_sigma=True)
plt.plot([0, 1], straight_line([0, 1], *popt), 'r-', label='fit')
plt.legend()
plt.ylabel('True home win prob.')
plt.xlabel('Est. home win prob. by odds')

# home goals
ax3 = plt.subplot(223, sharex=ax1)
plt.scatter(df_data['est_home_prob'], df_data['FTHG'], alpha=0.5)
plt.plot(df_grouped['est_home_prob'], df_grouped['FTHG'], 'k--', label='mean')
popt, _ = curve_fit(quadratic_fun, df_data['est_home_prob'], df_data['FTHG'], absolute_sigma=True)
xdata = df_data['est_home_prob'].sort_values()
plt.plot(xdata, quadratic_fun(xdata, *popt), 'r-', label='fit')
plt.legend()
plt.ylabel('Average home goals')
plt.xlabel('Est. away win prob. by odds')

# away win probability
ax2 = plt.subplot(222, sharey=ax1)
df_grouped = df_data[['est_away_prob', 'FTAG', 'away_win']].sort_values(by='est_away_prob').rolling(window=100, center=True).mean()
plt.scatter(df_grouped['est_away_prob'], df_grouped['away_win'], label='chunked mean', alpha=0.5)
plt.plot([0, 1], [0, 1], label='exact match')
popt_home, _ = curve_fit(straight_line, df_data['est_away_prob'], df_data['away_win'],
                         absolute_sigma=True)
plt.plot([0, 1], straight_line([0, 1], *popt_home), 'r-', label='fit')
plt.legend()
plt.ylabel('True away win prob.')
plt.xlabel('Est. away win prob. by odds')

# away goals
plt.subplot(224, sharex=ax2, sharey=ax3)
plt.scatter(df_data['est_away_prob'], df_data['FTAG'], alpha=0.5)
plt.plot(df_grouped['est_away_prob'], df_grouped['FTAG'], 'k--', label='mean')

popt_away, _ = curve_fit(quadratic_fun, df_data['est_away_prob'], df_data['FTAG'],
                         absolute_sigma=True)
xdata = df_data['est_away_prob'].sort_values()
plt.plot(xdata, quadratic_fun(xdata, *popt), 'r-', label='fit')
plt.legend()
plt.ylabel('Average away goals')
plt.xlabel('Est. away win prob. by odds')
plt.show()

df_grouped_mean = df_data[['est_draw_prob', 'draw']].sort_values(by='est_draw_prob').rolling(window=100, center=True).mean()
plt.plot(df_grouped_mean['est_draw_prob'], df_grouped_mean['draw'], label='mean')
popt_draw, _ = curve_fit(straight_line, df_data['est_draw_prob'], df_data['draw'],
                         absolute_sigma=True)
plt.plot([0, 0.5], straight_line([0, 0.5], *popt_draw), 'r-', label='fit')
plt.plot([0, 0.5], [0, 0.5], label='exact match')
plt.legend()
plt.ylabel('True draw prob.')
plt.xlabel('Est. draw prob. by odds')
plt.show()

ax1 = plt.subplot(211)
df_grouped = df_data[['est_home_away_diff', 'home_win', 'goal_diff']].sort_values(by='est_home_away_diff').rolling(window=100, center=True).mean()
plt.scatter(df_grouped['est_home_away_diff'], df_grouped['home_win'], label='mean', alpha=0.5)
popt, _ = curve_fit(straight_line, df_data['est_home_away_diff'], df_data['home_win'], absolute_sigma=True)
plt.plot([-1, 1], straight_line([-1, 1], *popt), 'r-', label='fit')
plt.legend()
plt.ylabel('Home win prob.')
plt.xlabel('est_home_away_diff')

ax3 = plt.subplot(212, sharex=ax1)
df_data1 = df_data.loc[(df_data['est_home_prob'] > df_data['est_draw_prob']) &
                       df_data['est_home_prob'] > df_data['est_away_prob'], :]
df_data2 = df_data.loc[(df_data['est_away_prob'] > df_data['est_draw_prob']) &
                       df_data['est_away_prob'] > df_data['est_home_prob'], :]
plt.scatter(df_data1['est_home_away_diff'], df_data1['goal_diff'], c='r', alpha=0.5)
plt.scatter(df_data2['est_home_away_diff'], df_data2['goal_diff'], c='b', alpha=0.5)
plt.plot(df_grouped['est_home_away_diff'], df_grouped['goal_diff'], 'k--', label='chunked mean')
popt, _ = curve_fit(straight_line, df_data['est_home_away_diff'], df_data['goal_diff'], absolute_sigma=True)
plt.plot([-1, 1], straight_line([-1, 1], *popt), 'r-', label='fit')
plt.legend()
plt.ylabel('Goal difference')
plt.xlabel('Est. away win prob. by odds')
plt.show()

df_data['sum_est_probs'] = df_data['est_home_prob'] + df_data['est_away_prob'] + df_data['est_draw_prob']

df_data['calc_home_prob'] = straight_line(df_data['est_home_prob'], *popt_home)
df_data['calc_away_prob'] = straight_line(df_data['est_away_prob'], *popt_home)
df_data['calc_draw_prob'] = straight_line(df_data['est_draw_prob'], *popt_home)

df_data['sum_calc_probs'] = df_data['calc_home_prob'] + df_data['calc_away_prob'] + df_data['calc_draw_prob']

plt.figure(figsize=(18, 6))
for league in ['I1', 'D1', 'E0', 'SP1']:
    df_grouped = df_data.loc[df_data.league == league, ['est_home_prob', 'est_away_prob', 'FTHG', 'FTAG']]\
        .sort_values(by=['est_home_prob', 'est_away_prob']).rolling(window=100, center=True).mean().fillna(method='bfill').fillna(method='ffill')
    for label in ['FTHG', 'FTAG']:
        x = df_grouped['est_home_prob']
        y = df_grouped['est_away_prob']
        z = df_grouped[label]

        # target grid to interpolate to
        xi = np.arange(min(x), max(x), 0.001)
        yi = np.arange(min(y), max(y), 0.001)
        xi, yi = np.meshgrid(xi, yi)

        # interpolate
        zi = griddata((x, y), z, (xi, yi), method='linear')

        # plot
        if label == 'FTHG':
            if league == 'I1':
                plt.subplot(241)
            if league == 'D1':
                plt.subplot(242)
            if league == 'E0':
                plt.subplot(243)
            if league == 'SP1':
                plt.subplot(244)
        else:
            if league == 'I1':
                plt.subplot(245)
            if league == 'D1':
                plt.subplot(246)
            if league == 'E0':
                plt.subplot(247)
            if league == 'SP1':
                plt.subplot(248)
        plt.contourf(xi, yi, zi, np.arange(-0.5, 4.5, 1))
        plt.xlabel('est_home_prob', fontsize=16)
        plt.ylabel('est_away_prob', fontsize=16)
        plt.title(label + ' | ' + league)
        cbar = plt.colorbar()
plt.show()


def linear_2d_model(X, k2_x, k2_y, k_x, k_y, offset):
    x, y = X
    return k2_x * x ** 2 + k_x * x + k2_y * y ** 2 + k_y * y + offset


plt.figure(figsize=(12, 6))
df_d1e0 = df_data.loc[(df_data.league == 'D1') | (df_data.league == 'E0'), :]
df_grouped = df_data.loc[(df_data.league == 'D1') | (df_data.league == 'E0'), ['est_home_prob', 'est_away_prob', 'FTHG', 'FTAG']] \
    .sort_values(by=['est_home_prob', 'est_away_prob']).rolling(window=100, center=True).mean().fillna(method='bfill').fillna(method='ffill')

for label in ['FTHG', 'FTAG']:
    for mode in ['data', 'model']:
        x = df_grouped['est_home_prob']
        y = df_grouped['est_away_prob']
        if mode == 'data':
            z = df_grouped[label]
        else:
            popt, _ = curve_fit(linear_2d_model, (df_d1e0['est_home_prob'], df_d1e0['est_away_prob']),
                                df_d1e0[label].values)
            z = linear_2d_model((x, y), *popt)
            pred_fit = linear_2d_model((df_d1e0['est_home_prob'], df_d1e0['est_away_prob']), *popt)
            r2_fit = np.round(r2_score(df_d1e0[label].values, pred_fit), 3)
            mse_fit = np.round(np.sqrt(mean_squared_error(df_d1e0[label].values, pred_fit)), 3)

        # target grid to interpolate to
        xi = np.arange(min(x), max(x), 0.001)
        yi = np.arange(min(y), max(y), 0.001)
        xi, yi = np.meshgrid(xi, yi)

        # interpolate
        zi = griddata((x, y), z, (xi, yi), method='linear')

        # plot
        if label == 'FTHG':
            if mode == 'data':
                plt.subplot(221)
            else:
                plt.subplot(222)
        else:
            if mode == 'data':
                plt.subplot(223)
            else:
                plt.subplot(224)
        plt.contourf(xi, yi, zi, np.arange(-0.5, 4.5, 1))
        plt.xlabel('est_home_prob', fontsize=16)
        plt.ylabel('est_away_prob', fontsize=16)
        if mode == 'data':
            plt.title(label + ' | ' + mode)
        else:
            plt.title(label + ' | ' + mode + '(' + str(r2_fit) + ' | ' + str(mse_fit) + ')')
        cbar = plt.colorbar()
plt.show()
