import time
import tensorflow as tf
import numpy as np
import pandas as pd

filename = "nba.games.stats.csv"

# Loads features from file and returns features dict
def load_data():
    df = pd.read_csv(filename)
    return

# load 2014-2015 stats for each team
# for each team all averages
def load_14_15_season():
    games = 82
    teams = 30
    df = pd.read_csv(filename)
    df_1415 = pd.DataFrame().reindex_like(df[0:teams])
    df_1415.drop(columns=["Unnamed: 0","Game","Date","Home", "Opponent", "WINorLOSS"], axis=1, inplace=True)
    columns = list(df_1415)[2:]
    df_row = 0
    df_1415_row = 0
    while df_1415_row < teams:

        df_1415.loc[df_1415_row, "Team"] = df.loc[df_row, "Team"]
        for column in columns:
            df_1415.loc[df_1415_row, column] = sum(df.loc[df_row:df_row+games,column])/games

        df_1415_row+=1
        df_row+=games

    print df_1415
    return


if __name__ == "__main__":
    print 'Hello'
    load_14_15_season()
