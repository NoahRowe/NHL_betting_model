#########################################################################################################
# IMPORTS
#########################################################################################################

import pandas as pd
import numpy as np


#########################################################################################################
# FEATURE FUNCTIONS
#########################################################################################################

def get_mean_game_features(last_game, feature_cols):
    return np.array([last_game[c] for c in feature_cols])


def get_diff_game_features(last_game, feature_cols, num_features):

    return np.array([last_game[feature_cols[k]] - last_game[feature_cols[k].replace("For", "Against")] 
                     if "For" in feature_cols[k] else last_game[feature_cols[k]]
                     for k in range(num_features)])


def loop_lookback_games(data, feature_cols, num_features, lookback_games, last_game, diff_feats=False):
    
    past_games = np.zeros((lookback_games, num_features if diff_feats else len(feature_cols)))
    
    last_game_id = last_game['last_game_id']
    
    # Loop through last games
    for j in range(lookback_games):

        # Check that we have a past game to look at
        if last_game_id != 0:

            # Get info for this game
            last_game_lines = data[data.gameId==last_game_id]

            last_game_was_home = last_game['was_last_home']
            if last_game_was_home:
                last_game = last_game_lines[last_game_lines.home_or_away=="HOME"].iloc[0]
            else:
                last_game = last_game_lines[last_game_lines.home_or_away=="AWAY"].iloc[0]

            past_games[j] = get_diff_game_features(last_game, feature_cols, num_features) \
                            if diff_feats else get_mean_game_features(last_game, feature_cols)

            # Get the new last game id for next loop
            last_game_id = last_game['last_game_id']

        else:
            break 
    
    return past_games


#########################################################################################################
# DATASET FUNCTIONS
#########################################################################################################

def single_team_data(data, feature_cols, num_features, lookback_games, diff_feats):

    X, Y, game_ids, seasons = [], [], [], []

    for i, row in data.iterrows():
        
        last_game = row
        
        past_games = loop_lookback_games(data=data, 
                                              feature_cols=feature_cols, 
                                              num_features=num_features, 
                                              lookback_games=lookback_games, 
                                              last_game=last_game, 
                                              diff_feats=diff_feats)

        # If any values are not filled, do not add to the array.
        if (past_games.sum(axis=1)==0).any():
            continue

        X.append(past_games)
        Y.append(row['team_won'])
        game_ids.append(row['gameId'])
    
    # Take average of X across past games dimension
    X = np.array(X).mean(axis=1)
    Y = np.array(Y)
    game_ids = np.array(game_ids)

    return X, Y, game_ids

def double_team_data(data, feature_cols, num_features, lookback_games, diff_feats):
    
    X, Y, game_ids = [], [], []
    
    for game_id in data.gameId.unique():
        
        game_lines = data[data.gameId==game_id]
        home_line = game_lines[game_lines.home_or_away=="HOME"].iloc[0]
        away_line = game_lines[game_lines.home_or_away=="AWAY"].iloc[0]
        
        # Get past games for home team
        home_past_games = loop_lookback_games(data=data, 
                                              feature_cols=feature_cols, 
                                              num_features=num_features, 
                                              lookback_games=lookback_games, 
                                              last_game=home_line, 
                                              diff_feats=diff_feats)
        
        # Get past games for away team
        away_past_games = loop_lookback_games(data=data, 
                                              feature_cols=feature_cols, 
                                              num_features=num_features, 
                                              lookback_games=lookback_games, 
                                              last_game=away_line, 
                                              diff_feats=diff_feats)
                
        # If any values are not filled, do not add to the array.
        if (away_past_games.sum(axis=1)==0).any() or (home_past_games.sum(axis=1)==0).any():
            continue
            
        # Take the mean of the differences we have
        home_past_games = home_past_games.mean(axis=0)
        away_past_games = away_past_games.mean(axis=0)
        
        # Take the differences of the means (for all columns)
        past_games_features = home_past_games - away_past_games
            
        X.append(past_games_features)
        Y.append(home_line['home_won'])
        game_ids.append(game_id)
        
    X = np.array(X)
    Y = np.array(Y)
    game_ids = np.array(game_ids)
        
    return X, Y, game_ids


#########################################################################################################
# ODDS FUNCTIONS
#########################################################################################################

def load_odds_data():
    '''
    Function to load in raw odds data.
    '''
    
    filenames = ["/home/nrowe/other/hockey/data/data/odds_data/nhl_odds_{:02d}-{:02d}.xlsx".format(i, i+1) 
                 for i in range(7, 22)]
    odds = [pd.read_excel(f) for f in filenames]
    
    return odds

def process_odds_data(odds):
    '''
    Function to process all odds data into a single dataset.
    '''
    
    seasons = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
    for i in range(len(odds)):
        odds[i] = odds[i].rename(columns={"OpenOU":"Open OU", "CloseOU":"Close OU"})
        if "PuckLine" in odds[i].columns:
            odds[i] = odds[i].drop(columns=["PuckLine", "Unnamed: 11"])
            odds[i] = odds[i].rename(columns={"Unnamed: 13":"Open OU odds", "Unnamed: 15":"Close OU odds"})
        elif "Puck Line" in odds[i].columns:
            odds[i] = odds[i].drop(columns=["Puck Line", "Unnamed: 11"])
            odds[i] = odds[i].rename(columns={"Unnamed: 13":"Open OU odds", "Unnamed: 15":"Close OU odds"})
        else:
            odds[i] = odds[i].rename(columns={"Unnamed: 11":"Open OU odds", "Unnamed: 13":"Close OU odds"})
        season = seasons[i:i+2]
        odds[i]["Date"] = [str(d) if len(str(d))==4 else "0"+str(d) for d in odds[i]["Date"]]
        odds[i]["Date"] = [str(season[0])+d if int(d)>800 else str(season[1])+d for d in odds[i]["Date"]]
        odds[i]["Date"] = [pd.to_datetime(d, format="%Y%m%d") for d in odds[i]["Date"]]

    odds = pd.concat(odds)
    
    # Fix the way the team names are represented
    odds['Team'] = [name.replace(" ","") for name in odds['Team']]
    oldNames = ['Montreal', 'Toronto', 'NYRangers', 'Chicago', 'Vancouver',
           'Calgary', 'SanJose', 'LosAngeles', 'Ottawa', 'Buffalo',
           'Winnipeg', 'Boston', 'Philadelphia', 'TampaBay', 'Edmonton',
           'St.Louis', 'Carolina', 'Nashville', 'Pittsburgh', 'Dallas',
           'Minnesota', 'Colorado', 'NewJersey', 'Columbus', 'NYIslanders',
           'Detroit', 'Arizona', 'Florida', 'Washington', 'Anaheim', 
           'Phoenix', "Atlanta", "WinnipegJets", "Vegas", "Arizonas", "Tampa", "SeattleKraken"]

    newNames = ["MTL", "TOR", "NYR", "CHI", "VAN", "CGY", "SJS", "LAK", "OTT", "BUF", "WPG", "BOS", "PHI", "TBL", 
                "EDM", "STL", "CAR", "NSH", "PIT", "DAL", "MIN", "COL", "NJD", "CBJ", "NYI", "DET", "ARI", "FLA", 
                "WSH", "ANA", "ARI", "ATL", "WPG", "VGK", "ARI", "TBL", "SEA"]
    odds = odds.replace(oldNames, newNames) 
    odds.reset_index(drop=True, inplace=True)

    return odds

def combine_odds_data(odds, data):
    '''
    Function to combine the raw MP data and the processed odds.
    '''

    def US2Dec(x):
        return 1+(int(x)/100) if int(x)>0 else 1-(100/int(x))

    if "odds" not in data.columns:
        data['odds'] = np.nan
        data['imp_prob'] = np.nan
    
    odds_type = "Close"
    for i in range(len(data)):
        game = data.iloc[i]
        date = game['gameDate']
        team = game['team']
        
        if not np.isnan(game["odds"]):
            continue

        # Get the odds value
        val = odds[(odds.Date==date) & (odds.Team==team)]
        if len(val)>1:
            print(val)
            raise ValueError("Odds Data: More than one team loaded.")
        # Conver to decimal
        elif len(val)==1:
            try:
                data["odds"].iloc[i] = US2Dec(val[odds_type].values[0])
                data["imp_prob"].iloc[i] = 1/US2Dec(val[odds_type].values[0])
            except:
                print("Odds Processing Errored.", date.strftime('%Y-%m-%d'), team, i)
        else:
            print("No odds data found for:", date.strftime('%Y-%m-%d'), team)
            
    return data

def combine_OU_odds_data(odds, data):
    '''
    Function to combine the raw mp data and the processed odds.
    '''

    def US2Dec(x):
        return 1+(int(x)/100) if int(x)>0 else 1-(100/int(x))

    if "OU_odds" not in data.columns:
        data['OU_odds'] = np.nan
        data['OU_imp_prob'] = np.nan
        data['OU_line'] = np.nan
    
    odds_type = "Close OU odds"
    line_value = "Close OU"
    for i in range(len(data)):
        game = data.iloc[i]
        date = game['gameDate']
        team = game['team']
        
        if not np.isnan(game["OU_odds"]):
            continue

        # Get the odds value
        val = odds[(odds.Date==date) & (odds.Team==team)]
        if len(val)>1:
            print(val)
            raise ValueError("Odds Data: More than one team loaded.")

        elif len(val)==1:
            try:
                data["OU_odds"].iloc[i] = US2Dec(val[odds_type].values[0])
                data["OU_imp_prob"].iloc[i] = 1/US2Dec(val[odds_type].values[0])
                data["OU_line"].iloc[i] = val[line_value].values[0]
            except:
                print("Odds Processing Errored.", date.strftime('%Y-%m-%d'), team, i)
        else:
            print("No odds data found for:", date.strftime('%Y-%m-%d'), team)
            
    return data