import numpy as np
import math


def select_team_stats_from_game_data(df_games, stat_label, team_name):
    home_stat_data = df_games.loc[df_games['HomeTeam'] == team_name, [stat_label + '_home']]
    home_stat_data = home_stat_data.rename(columns={stat_label + '_home': stat_label})

    away_stat_data = df_games.loc[df_games['AwayTeam'] == team_name, [stat_label + '_away']]
    away_stat_data = away_stat_data.rename(columns={stat_label + '_away': stat_label})

    df_stat_values = home_stat_data.append(away_stat_data)
    return df_stat_values.sort_index(inplace=False)


def map_team_stats_to_game_data(df_games, df_stat_values, stat_label, team_name):
    home_ids = df_games.loc[df_games['HomeTeam'] == team_name].index
    away_ids = df_games.loc[df_games['AwayTeam'] == team_name].index

    home_stat_data = df_stat_values.loc[df_stat_values.index.isin(set(home_ids)), [stat_label]]
    df_games.loc[df_games.index.isin(set(home_ids)), stat_label + '_home'] = home_stat_data[stat_label].values

    away_stat_data = df_stat_values.loc[df_stat_values.index.isin(set(away_ids)), [stat_label]]
    df_games.loc[df_games.index.isin(set(away_ids)), stat_label + '_away'] = away_stat_data[stat_label].values
    return df_games.sort_index(inplace=False)


def shift_stats_by_team(df_games, stat_label):
    if df_games.index.name != 'game_id':
        df_games.set_index('game_id', inplace=True, drop=True)

    for team_name in df_games['HomeTeam'].unique():
        df_stat_for_team = select_team_stats_from_game_data(df_games, stat_label, team_name)
        # shift data that for the current game only historic data is used and not the game itself
        df_stat_for_team[stat_label] = np.append([df_stat_for_team[stat_label].values[0]],
                                                 df_stat_for_team[stat_label].values[:-1])
        df_games = map_team_stats_to_game_data(df_games, df_stat_for_team, stat_label, team_name)
    return df_games.reset_index(inplace=False, drop=False)


def smooth_stats_by_team(df_games, stat_label, win_length=17, win_type=None):
    if df_games.index.name != 'game_id':
        df_games.set_index('game_id', inplace=True, drop=True)

    for team_name in df_games['HomeTeam'].unique():
        df_stat_for_team = select_team_stats_from_game_data(df_games, stat_label, team_name)

        # convert to a dataframe to use the rolling mean function
        if df_stat_for_team.shape[0] < win_length:
            rolling_window = int(math.floor(df_stat_for_team.shape[0] / 2)) * 2
            if rolling_window == 0:
                continue
        else:
            rolling_window = win_length
        df_stat_for_team[stat_label] = df_stat_for_team[stat_label].rolling(
            window=rolling_window,
            min_periods=int(math.ceil(rolling_window / 2)),
            center=False,
            win_type=win_type).mean()
        # fill nan values at the edges of the dataframe
        df_stat_for_team[stat_label].fillna(method='bfill', inplace=True)
        df_stat_for_team[stat_label].fillna(method='ffill', inplace=True)

        df_games = map_team_stats_to_game_data(df_games, df_stat_for_team, stat_label, team_name)
    return df_games.reset_index(inplace=False, drop=False)


def get_most_recent_stat(df_games, stat_label, team_name, game_date=None):
    if game_date is None:
        game_date = df_games.datetime.max()
    df_team = df_games.loc[((df_games.HomeTeam == team_name) |
                            (df_games.AwayTeam == team_name)) &
                           (df_games.datetime < game_date)]
    if df_team.shape[0] == 0:
        return 0
    most_recent = df_team.loc[
        df_team.game_id == df_team.game_id.max()]
    if most_recent.HomeTeam.values[0] == team_name:
        team_stat = most_recent[stat_label + '_home'].values[0]
    else:
        team_stat = most_recent[stat_label + '_away'].values[0]
    return team_stat
