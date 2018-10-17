import time
import tensorflow as tf
import numpy as np
import pandas as pd


# Loads features from file and returns features dict
def load_data(filename):
    df = pd.read_csv(filename)
    return df

# load 2014-2015 stats for each team
def create_14_15_team_csv():
    games = 82
    teams = 30
    df = pd.read_csv("nba.games.stats.csv")
    df_1415 = pd.DataFrame().reindex_like(df[0:teams])
    df_1415.drop(columns=["Unnamed: 0","Game","Date","Home", "Opponent", "WINorLOSS"], axis=1, inplace=True)
    columns = list(df_1415)[1:]
    df_row = 0
    df_1415_row = 0
    while df_1415_row < teams:

        df_1415.loc[df_1415_row, "Team"] = df.loc[df_row, "Team"]
        for column in columns:
            df_1415.loc[df_1415_row, column] = sum(df.loc[df_row:df_row+games,column])/float(games)

        df_1415_row+=1
        df_row+=games

    df_1415.to_csv("nba.team.1415.csv")
    return

def create_14_15_season_csv():
    df = pd.read_csv("nba.games.stats.csv")
    # only leave 14 15 season
    # remove every column but team, home, opponent, WINorLoss

if __name__ == "__main__":
    print 'Hello'
    # Creates csv for average team stats
    #create_14_15_team_csv()

    # Figure out most important stats
    df = load_data("nba.team.1415.csv")
    #np.histogram(np.array(df["TeamPoints"]))

    # Binning data teampoints next by 3 points

    # Predict nba game outcome, predict score of each team
    # Two teams - each team has stats - percentages
    # Training set - 50 games from each team, feature the one team stats vs other, target 1 or 2
    # Testing set - 10 games from each team,
    # Validation - 10 games from each team

    #feature - team 1 stats, team 1 location, team 2 stats, target - team 1  w/l