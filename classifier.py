import time
import tensorflow as tf
import numpy as np
import gzip
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.python.data import Dataset

games = 82
teams = 30

# load 2014-2015 stats for each team
def create_14_15_team_csv():
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

# Creates a table of all match ups and result of 14/15 season
def create_14_15_season_csv():
    df = pd.read_csv("nba.games.stats.csv")
    keep_columns = ["Team", "Game", "Home", "Opponent", "WINorLOSS"]
    df = df[keep_columns]
    df = df[:games*teams]
    df.to_csv("nba.season.1415.csv")
    # only leave 14 15 season
    # remove every column but team, home, opponent, WINorLoss

# Feature - team 1 stats, team 1 location target - team 1  w/l
def create_raw_features_csv():
    print 'making feature'
    pre ="v."
    df_keys = pd.read_csv("nba.season.1415.csv")
    df_values = pd.read_csv("nba.team.1415.csv")
    #opp_df_values = df_values.rename(columns=lambda col_name: pre+col_name)

    df = df_keys.merge(df_values, on=["Team"])
    #df = df.merge(opp_df_values, left_on=["Opponent"], right_on=[pre+"Team"])

    df.drop(columns=["Unnamed: 0_x", "Unnamed: 0_y"], axis=1, inplace=True)
    #df.drop(columns=[pre+"Unnamed: 0", pre+"Team"], axis=1, inplace=True)
    df.to_csv("raw_features.csv")

# Fix features so they are ready for machine learning
def feature_engineering():
    df = pd.read_csv("raw_features.csv")

    # Convert all teams to numbers
    team_names = df["Team"].unique()
    team_names_dict = dict(zip(team_names, range(len(team_names))))
    df = df.replace(team_names_dict)

    # Convert Home to 1 or 0
    df["Home"] = pd.get_dummies(df["Home"])["Home"]

    # Convert WinLoss to 1 or 0
    df['WINorLOSS'] = pd.get_dummies(df['WINorLOSS'])['W']
    df.drop(columns=["Unnamed: 0"], axis=1, inplace=True)

    corr = df.corr()['WINorLOSS']
    for key, value in corr.items():
        if (abs(value) < 0.1):
            df.drop([key], axis=1, inplace=True)

    df.to_csv("features.csv")

# Loads features and labels
def load_features_labels():
    data = pd.read_csv("features.csv")
    labels = data["WINorLOSS"]
    data.drop("WINorLOSS", axis=1, inplace=True)
    return {"data": data}, labels

# Input function used with dnn_classifier returns iterators of features, labels
def input_function(features, targets, batch_size=1, shuffle=True, num_epochs=None):

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    # Return the next batch of data
    feature, label = ds.make_one_shot_iterator().get_next()
    return feature, label


if __name__ == "__main__":

    print 'Hello'

    # Creates csv for average team stats
    #create_14_15_team_csv()

    # Creates csv for season keys
    #create_14_15_season_csv()

    # Creates features csv
    #create_raw_features_csv()

    # Fix feature values
    #feature_engineering()


    features, labels = load_features_labels()



    # Binning data teampoints next by 3 points

    # Predict nba game outcome, predict score of each team
    # Two teams - each team has stats - percentages
    # Training set - 50 games from each team, feature the one team stats vs other, target 1 or 2
    # Testing set - 10 games from each team,
    # Validation - 10 games from each team