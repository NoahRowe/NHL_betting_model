#!/usr/bin/env python3

import numpy as np

import requests as r
import pickle

from pathlib import Path

game_type = "02" # Regular season
max_game_num = 1500 # Never more games then this


years = np.arange(2008, 2023) # Will download placeholders for current year
for year in years:
    
    # Init game ids
    game_ids = [f"{year}{game_type}{i:04}" for i in range(1, max_game_num)]
    
    # Create directory for this year
    year_directory = f"{year}/"

    Path(year_directory).mkdir(exist_ok=True)

    for game_id in game_ids:
        url = f"https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live?site=en_nhl"
        game_data = r.get(url).json()

        # Check if we have reached the end
        if "message" in game_data.keys():
            print("No data found:", game_id)
            break

        # Save the json
        save_name = f"{year_directory}/pbp_{game_id}.pkl"
        with open(save_name, "wb") as f:
            pickle.dump(game_data, f)

    print(f"Done {year}.")
