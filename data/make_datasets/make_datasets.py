#!/usr/bin/env python3

#########################################################################################################
# IMPORTS
#########################################################################################################

import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler


from make_datasets_functions import load_odds_data, process_odds_data, combine_odds_data
from make_datasets_functions import single_team_data, double_team_data


#########################################################################################################
# LOAD IN THE DATA
#########################################################################################################

print("Loading in core data...")
data = pd.read_csv("/home/nrowe/other/hockey/data/data/downloaded_data/mp_data/all_teams.csv")
data = data[data.situation=="all"]
data = data[data.playoffGame==0]

data = data.sort_values(by=['gameId', 'home_or_away'])
data.reset_index(drop=True, inplace=True)

data["gameDate"] = pd.to_datetime(data["gameDate"], format='%Y%m%d')

data.replace({"T.B":"TBL", "L.A":"LAK", "N.J":"NJD", "S.J":"SJS"}, inplace=True)
print("Done.")

#########################################################################################################
# ADD ODDS
#########################################################################################################

print("Adding odds data...")
odds_data = load_odds_data()
odds_data = process_odds_data(odds_data)
data = combine_odds_data(odds_data, data)
print("Done.")


#########################################################################################################
# CLEAN THE DATA
#########################################################################################################

# Remove game ids that dont have both a home and away
bad_game_ids = []
for game_id in data.gameId.unique():
    game_lines = data[data.gameId==game_id]
    if game_lines[game_lines.home_or_away=="HOME"].shape[0] != 1:
        bad_game_ids.append(game_id)
        continue
    elif game_lines[game_lines.home_or_away=="AWAY"].shape[0] != 1:
        bad_game_ids.append(game_id)
        continue
        
data = data[~data.gameId.isin(bad_game_ids)]

# Add a last game played column (and whether they are home or away)
new_cols = ['last_game_id', 'was_last_home']
for c in new_cols:
    data[c] = 0

# Do it by team subset
teams = data.team.unique()

for team in teams:
    team_subset = data[data.team==team]
    
    team_game_ids = team_subset.gameId.unique()
    
    for i in range(1, len(team_game_ids)):
        game_line = team_subset[team_subset.gameId==team_game_ids[i]]
        was_home = team_subset[team_subset.gameId==team_game_ids[i-1]].home_or_away.iloc[0] == "HOME"

        for index in game_line.index:
            data.at[index, 'last_game_id'] = int(team_game_ids[i-1])
            data.at[index, 'was_last_home'] = was_home
            
            
#########################################################################################################
# LOAD PLAY BY PLYA
#########################################################################################################

print("Loading play-by-play data...")
pbp_data = dict()
for season in data.season.unique():
    season_data = data[data.season==season]
    
    season_dict = dict()
    
    for game_id in season_data.gameId.unique():
        
        path = f"../main_data/pbp_data/{season}/pbp_{str(game_id)}.pkl"
        with open(path, "rb") as f:
            temp_dict = pickle.load(f)['liveData']
            
        # Get who won
        winning_goalie = temp_dict['decisions']['winner']['id']
        home_won = winning_goalie in temp_dict['boxscore']['teams']['home']['goalies']

        # Add to dictionary
        season_dict[game_id] = home_won
    
    pbp_data[season] = season_dict

print("Done.")

# Add a home_won column
data['home_won'] = data.apply(lambda x: pbp_data[x['season']][x['gameId']], axis=1)
# Add a team_won column
data['team_won'] = data.apply(lambda x: x['home_won'] if x['home_or_away']=="HOME" else not x['home_won'], axis=1)


#########################################################################################################
# DEFINE FEATURE COLUMNS
#########################################################################################################

feature_cols = [
    'xGoalsPercentage', 'corsiPercentage', 'fenwickPercentage', 'xOnGoalFor', 'xReboundsFor', 
    'xPlayContinuedInZoneFor', 'flurryAdjustedxGoalsFor' , 'shotsOnGoalFor', 'missedShotsFor', 
    'blockedShotAttemptsFor', 'goalsFor', 'reboundsFor', 'freezeFor', 'savedShotsOnGoalFor', 
    'penalityMinutesFor', 'faceOffsWonFor', 'hitsFor', 'takeawaysFor', 'giveawaysFor',
    'lowDangerxGoalsFor', 'mediumDangerxGoalsFor', 'highDangerxGoalsFor', 'scoreAdjustedShotsAttemptsFor', 
    'scoreAdjustedUnblockedShotAttemptsFor', 'dZoneGiveawaysFor', 'reboundxGoalsFor', 'totalShotCreditFor'
]

num_features = len(feature_cols)
# Create against columns
for col in feature_cols:
    if "For" in col:
        feature_cols.append(col.replace("For", "Against"))


#########################################################################################################
# DEFINE LOOP PARAMETERS
#########################################################################################################

lookback_games = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# lookback_games = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]


#########################################################################################################
# CREATE AND SAVE THE DATA
#########################################################################################################

save_path = "../main_data/feature_data/"

# Save the master datafile first
data.to_pickle("../main_data/master_data.df")

# Loop and save the individual files
for lookback_game in lookback_games:
    
    # Make both types of single team data
    single_mean_X, single_mean_Y, single_mean_ids = single_team_data(data=data, 
                                                                     feature_cols=feature_cols, 
                                                                     num_features=num_features, 
                                                                     lookback_games=lookback_game, 
                                                                     diff_feats=False)
    
    single_diff_X, single_diff_Y, single_diff_ids = single_team_data(data=data, 
                                                                     feature_cols=feature_cols, 
                                                                     num_features=num_features, 
                                                                     lookback_games=lookback_game, 
                                                                     diff_feats=True)
    
    # Make both types of two team data
    double_mean_X, double_mean_Y, double_mean_ids = double_team_data(data=data, 
                                                                     feature_cols=feature_cols, 
                                                                     num_features=num_features, 
                                                                     lookback_games=lookback_game, 
                                                                     diff_feats=False)
    
    double_diff_X, double_diff_Y, double_diff_ids = double_team_data(data=data, 
                                                                     feature_cols=feature_cols, 
                                                                     num_features=num_features, 
                                                                     lookback_games=lookback_game, 
                                                                     diff_feats=True)
    
#     Save the data
    np.save(save_path + f"single_mean_lg{lookback_game}_X.npy", single_mean_X)
    np.save(save_path + f"single_mean_lg{lookback_game}_Y.npy", single_mean_Y)
    np.save(save_path + f"single_mean_lg{lookback_game}_ids.npy", single_mean_ids)
    
    np.save(save_path + f"single_diff_lg{lookback_game}_X.npy", single_diff_X)
    np.save(save_path + f"single_diff_lg{lookback_game}_Y.npy", single_diff_Y)
    np.save(save_path + f"single_diff_lg{lookback_game}_ids.npy", single_diff_ids)
    
    np.save(save_path + f"double_mean_lg{lookback_game}_X.npy", double_mean_X)
    np.save(save_path + f"double_mean_lg{lookback_game}_Y.npy", double_mean_Y)
    np.save(save_path + f"double_mean_lg{lookback_game}_ids.npy", double_mean_ids)
    
    np.save(save_path + f"double_diff_lg{lookback_game}_X.npy", double_diff_X)
    np.save(save_path + f"double_diff_lg{lookback_game}_Y.npy", double_diff_Y)
    np.save(save_path + f"double_diff_lg{lookback_game}_ids.npy", double_diff_ids)
    
    print(f"Data for lookback_game = {lookback_game} saved.")