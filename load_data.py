import os
from datetime import datetime
from difflib import SequenceMatcher

import Levenshtein as Levenshtein
import requests
from os import path
import pandas as pd
import numpy as np
import json
from io import StringIO

COLUMNS_USED = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'IWH', 'IWD', 'IWA']


def try_parsing_date(text):
    for fmt in ('%d/%m/%y', '%d/%m/%Y'):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError('no valid date format found')


def load_from_football_data(league, year):
    base_url = 'https://www.football-data.co.uk/mmz4281/'
    data_url1 = base_url + year + '/' + league + '.csv'
    response = requests.get(data_url1)
    # account for the change of url over the years
    if (response.status_code == 404) or (response.status_code == 300):
        data_url2 = base_url + year[2:] + '{0:0=2d}'.format(int(year[2:]) + 1) + '/' + league + '.csv'
        response = requests.get(data_url2)
    if response.status_code != 200:
        print('No data found for ' + league + ' in year ' + year + ' on site ' + base_url)
        return None
    response_string = str(response.content, 'utf-8')
    data = StringIO(response_string)
    df_season = pd.read_csv(data, sep=",", usecols=COLUMNS_USED)
    df_season = df_season.loc[df_season.isna().sum(axis=1) == 0].reset_index(drop=True)
    df_season['datetime'] = [try_parsing_date(x) for x in df_season['Date']]
    if df_season['datetime'].min().to_period('Y').year != int(year):
        print('Missmatch in the season received (' + str(df_season['datetime'].min().to_period('Y').year) +
              ') and the season requested (' + year + ') for ' + league)
        return None
    df_season['season'] = year + '/' + str(int(year) + 1)
    # use 9 games per gameday as in the bundesliga (it is only used for analysis)
    games_per_gameday = 9
    num_gamedays = int(df_season['season'].shape[0] / games_per_gameday)
    df_season = df_season.loc[df_season.index < (num_gamedays * games_per_gameday), :]
    df_season['game_day'] = np.repeat(np.linspace(1, num_gamedays, num_gamedays), games_per_gameday).astype('int')
    df_season['league'] = league

    df_season['result'] = [str(int(a)) + '-' + str(int(b)) for a, b in
                           zip(df_season['FTHG'].values, df_season['FTAG'].values)]
    df_season['goal_diff'] = df_season['FTHG'] - df_season['FTAG']
    df_season['odds_diff'] = df_season['IWH'] - df_season['IWA']
    df_season['game_id'] = df_season['datetime'].astype('str') + df_season['HomeTeam'] + df_season['AwayTeam']
    if df_season.shape[0] == 0:
        print('No data for ' + league + ' in year ' + year)
        return None
    return df_season


def load_league_data_with_odds(seasons, leagues=None):
    date_today = datetime.now()
    if leagues is None:
        leagues = ['D1']
    data_files = [y + '-' + str(x) for x in seasons for y in leagues]
    df_seasons = pd.DataFrame()
    for file in data_files:
        [league, year] = str.split(file, '-')
        current_season = date_today.year if date_today.month > 7 else date_today.year - 1
        if int(year) == current_season:
            df_season = load_from_football_data(league, year)
            df_season.to_csv('data/' + file + '.csv', index=False)
        else:
            path_to_file = 'data/' + file + '.csv'
            if path.exists(path_to_file):
                df_season = pd.read_csv('data/' + file + '.csv')
                df_season['datetime'] = pd.to_datetime(df_season['datetime'], format="%Y-%m-%d")
            else:
                df_season = load_from_football_data(league, year)
                if df_season is None:
                    continue
                df_season.to_csv('data/' + file + '.csv', index=False)
        df_seasons = df_seasons.append(df_season, ignore_index=True)
    return_data = df_seasons.sort_values('datetime').drop_duplicates('game_id', keep='last').reset_index(drop=True)
    if return_data.shape != df_seasons.shape:
        print('There were duplicates in the loaded data -> investigate!')
    return return_data


def load_538_spi_data(url):
    response = requests.get(url)
    if response.status_code != 200:
        print('Could not get SPI from 538!')
        return None
    response_string = str(response.content, 'utf-8')
    data = StringIO(response_string)
    df_spi = pd.read_csv(data, sep=",")
    df_spi = df_spi.loc[(df_spi.league == 'Barclays Premier League') |
                        (df_spi.league == 'German Bundesliga') |
                        (df_spi.league == 'Spanish Primera Division') |
                        (df_spi.league == 'Italy Serie A'), :]
    df_spi['datetime'] = [datetime.strptime(text, '%Y-%m-%d') for text in df_spi['date']]
    if df_spi.shape[0] == 0:
        return None
    return df_spi


def load_latest_538_spi_data():
    url_538_latest = 'https://projects.fivethirtyeight.com/soccer-api/club/spi_matches_latest.csv'
    return load_538_spi_data(url_538_latest)


def load_all_538_spi_data():
    url_538_all = 'https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv'
    return load_538_spi_data(url_538_all)


def create_team_mapping(base_teams, teams_to_map, mapping_name):
    path_to_mapping = 'data/' + mapping_name + '.json'

    def calc_lev_mismatch(base_name, name_to_match):
        leven_dist = Levenshtein.distance(base_name, name_to_match)
        len_name = max([len(name_to_match), len(base_name)])
        return leven_dist / len_name

    if os.path.isfile(path_to_mapping):
        with open(path_to_mapping, 'r') as json_file:
            mapping_dict = json.load(json_file)
    else:
        mapping_dict = {}

    teams_to_map_unique = list(set(teams_to_map) - set(mapping_dict.keys()))
    base_team_unique = list(set(base_teams) - set(mapping_dict.values()))
    for candidate in teams_to_map_unique:
        if candidate in mapping_dict.keys():
            continue
        if len(base_team_unique) == 0:
            print('Mapping finished early -> not all team names are mapped')
            break
        low_candidate = candidate.lower()
        lev_mismatch = [calc_lev_mismatch(low_candidate, x.lower()) for x in base_team_unique]
        seq_match = [SequenceMatcher(None, low_candidate, x.lower()).ratio() for x in base_team_unique]
        idx_best = np.argmax(np.subtract(seq_match, lev_mismatch))
        mapping_dict[candidate] = base_team_unique[idx_best]
        base_team_unique = np.delete(base_team_unique, idx_best)
    json_dict = json.dumps(mapping_dict)
    f = open(path_to_mapping, 'w')
    f.write(json_dict)
    f.close()
    return mapping_dict


def load_odds_and_538_data(leagues=None):
    if leagues is None:
        leagues = ['D1']
    df_538 = load_all_538_spi_data()
    df_data = load_league_data_with_odds(range(2016, 2021), leagues)

    team_map = create_team_mapping(np.concatenate((df_data.HomeTeam, df_data.AwayTeam)),
                                   np.concatenate((df_538.team1, df_538.team2)), 'map538_to_odds')

    df_538['MappedHomeTeam'] = [team_map[x] for x in df_538['team1']]
    df_538['MappedAwayTeam'] = [team_map[x] for x in df_538['team2']]
    df_538['game_id'] = df_538['datetime'].astype('str') + df_538['MappedHomeTeam'] + df_538['MappedAwayTeam']

    columns_to_use = ['team1', 'team2', 'spi1', 'spi2', 'prob1', 'prob2', 'probtie', 'proj_score1', 'proj_score2',
                      'game_id', 'MappedHomeTeam', 'MappedAwayTeam']
    df_538_eval = df_538.loc[:, columns_to_use].set_index('game_id', inplace=False).join(
        df_data.loc[:, ['game_id', 'season', 'datetime', 'league', 'game_day', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG',
                        'IWH', 'IWD', 'IWA', 'goal_diff', 'odds_diff', 'result']
        ].set_index('game_id', inplace=False))

    return df_538_eval.loc[df_538_eval.isna().sum(axis=1) == 0]
