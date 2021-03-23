from scipy.optimize import curve_fit

from models.basic_math_functions import straight_line


def create_odds_probability_model(df_games):
    df_fit = df_games.copy(deep=True)

    df_fit['home_prob'] = 1/df_fit['IWH']
    df_fit['away_prob'] = 1/df_fit['IWA']

    popt_straight_home, _ = curve_fit(straight_line, df_fit['home_prob'], df_fit['FTHG'])
    popt_straight_away, _ = curve_fit(straight_line, df_fit['away_prob'], df_fit['FTAG'])

    def estimate_with_logistic_curve(iwh, iwa):
        home_score = straight_line(1/iwh, *popt_straight_home)
        away_score = straight_line(1/iwa, *popt_straight_away)
        return [home_score, away_score]

    return estimate_with_logistic_curve
