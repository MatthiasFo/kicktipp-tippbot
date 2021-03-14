import numpy as np
import pandas as pd
from robobrowser import RoboBrowser

from KickTippTipper.tipper import login, set_bet_urls, grab_kicktipp_groups, grab_odds, pass_results
from utils.generate_and_convert_picks import convert_pick_array_to_string
from load_data import load_odds_and_538_data, load_latest_538_spi_data, create_team_mapping
from models.model_dixon_cole_with_spi_odds import create_combined_model

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

df_538_latest = load_latest_538_spi_data()
df_data = load_odds_and_538_data(leagues=['D1', 'E0'])

combined_model = create_combined_model(df_data, 'D1')

browser = RoboBrowser(parser="html.parser", history=True)
login(browser)
betting_url = set_bet_urls(grab_kicktipp_groups(browser))
my_odds = grab_odds(browser, betting_url)

kt_teams = [x for key in my_odds for x in my_odds[key]['teams']]
df_bundesliga = df_538_latest.loc[df_538_latest.league == 'German Bundesliga']
team_map = create_team_mapping(np.concatenate((df_bundesliga.team1, df_bundesliga.team2)), kt_teams, 'map_kt_to_538')

df_kt_data = pd.DataFrame([{'team1': team_map[my_odds[key]['teams'][0]], 'team2': team_map[my_odds[key]['teams'][1]],
                            'odds_home': my_odds[key]['odds'][0], 'odds_draw': my_odds[key]['odds'][1],
                            'odds_away': my_odds[key]['odds'][2]}
                           for key in my_odds])

df_gameday = df_kt_data.set_index(['team1', 'team2']).join(
    df_538_latest[['team1', 'team2', 'proj_score1', 'proj_score2', 'spi1', 'spi2']].set_index(['team1', 'team2']),
    how='left', on=['team1', 'team2']).reset_index(drop=False)

# display the three picks with the highest expected reward | pick (exp. reward)
matches = []
for index, row in df_gameday.iterrows():
    inp_data = row[['team1', 'team2', 'odds_home', 'odds_draw', 'odds_away',
                    'spi1', 'spi2', 'proj_score1', 'proj_score2']].values
    pick = combined_model(*inp_data)
    df_gameday.loc[df_gameday.index == index, 'best pick'] = convert_pick_array_to_string(pick[0][0])
    df_gameday.loc[df_gameday.index == index, 'best value'] = np.round(pick[0][1], 3)
    df_gameday.loc[df_gameday.index == index, '2nd best pick'] = convert_pick_array_to_string(pick[1][0])
    df_gameday.loc[df_gameday.index == index, '2nd best value'] = np.round(pick[1][1], 3)
    matches.append(list(pick[0][0]))
print('------------------------------------------------------------------------------------------')
print(df_gameday)

pass_results(browser, betting_url, matches)
print('Tipp upload finished.')
