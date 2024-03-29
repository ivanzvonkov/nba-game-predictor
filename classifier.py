import os
import time
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from tensorflow.python.estimator.export import export

# CHANGE BELOW BASED ON PLAYOFF YEAR
# ----------------------------------------------------------------------------
season = '1718'
create_data = True
# ----------------------------------------------------------------------------

s = {
    '1415': 0,
    '1516': 1,
    '1617': 2,
    '1718': 3
}

games = 82
teams = 30

# load 2014-2015 stats for each team
def create_team_csv():
    df = pd.read_csv("nba.games.stats.csv")
    df_teams = pd.DataFrame().reindex_like(df[0:teams])
    df_teams.drop(columns=["Unnamed: 0","Game","Date","Home", "Opponent", "WINorLOSS"], axis=1, inplace=True)
    columns = list(df_teams)[1:]
    df_row = games*teams*s[season]
    df_team_row = 0
    while df_team_row < teams:
        df_teams.loc[df_team_row, "Team"] = df.loc[df_row, "Team"]
        for column in columns:
            df_teams.loc[df_team_row, column] = sum(df.loc[df_row:df_row+games,column])/float(games)

        df_team_row+=1
        df_row+=games

    df_teams.to_csv(season+"_season/nba.team.csv")
    return

# Creates a table of all match ups and result of specified season
def create_season_csv():
    df = pd.read_csv("nba.games.stats.csv")
    keep_columns = ["Team", "Game", "Home", "Opponent", "WINorLOSS"]
    df = df[keep_columns]
    start = games*teams*s[season]
    end = start + (games*teams)
    df = df[start :end]
    df.to_csv(season+"_season/nba.season.csv")

# Feature - team 1 stats, team 1 location target - team 1  w/l
def create_raw_features_csv():
    print 'making feature'
    pre ="v."
    df_keys = pd.read_csv(season+"_season/nba.season.csv")
    df_values = pd.read_csv(season+"_season/nba.team.csv")
    opp_df_values = df_values.rename(columns=lambda col_name: pre+col_name)

    df = df_keys.merge(df_values, on=["Team"])
    df = df.merge(opp_df_values, left_on=["Opponent"], right_on=[pre+"Team"])

    df.drop(columns=["Unnamed: 0_x", "Unnamed: 0_y"], axis=1, inplace=True)
    df.drop(columns=[pre+"Unnamed: 0", pre+"Team"], axis=1, inplace=True)
    df.to_csv(season+"_season/raw_features.csv")

# Fix features so they are ready for machine learning
def feature_engineering():
    df = pd.read_csv(season+"_season/raw_features.csv")

    # Convert Home to 1 or 0
    #df["Home"] = pd.get_dummies(df["Home"])["Home"]

    # Remove all non home games (removes duplicates) and delete home column
    df = df[df.Home == "Home"]
    # Drop home column
    df.drop("Home", axis=1, inplace=True)

    # Convert WinLoss to 1 or 0
    df['WINorLOSS'] = pd.get_dummies(df['WINorLOSS'])['W']
    df.drop(columns=["Unnamed: 0"], axis=1, inplace=True)

    # Drop if correlation is less than 0.1
    corr = df.corr()['WINorLOSS']
    skip_rows = ['Team', 'Opponent']
    for key, value in corr.items():
        if (abs(value) < 0.1) and (key not in skip_rows):
            df.drop([key], axis=1, inplace=True)

    # Scale down values
    skip_rows = ['Unnamed: 0','Team','Game','Opponent','WINorLOSS']
    for column in df:
        if column not in skip_rows:
            median = df[column].median()
            std = df[column].std()
            df[column] = df[column].apply(lambda val: (val-median)/std)

    # Convert all teams to numbers
    # team_names = df["Team"].unique()
    # team_names_dict = dict(zip(team_names, range(len(team_names))))
    # df = df.replace(team_names_dict)

    # Convert teams to one hot encoding
    new_teams = pd.get_dummies(data=df["Team"], prefix='h.')
    new_opps = pd.get_dummies(data=df["Opponent"], prefix='a.')
    df.drop("Team", axis=1, inplace=True)
    df.drop("Opponent", axis=1, inplace=True)
    df = pd.concat([df, new_teams, new_opps], axis=1, sort=False)

    df.to_csv(season+"_season/features.csv")

# Loads features and labels
def load_features_labels():
    features = pd.read_csv(season+"_season/features.csv").sample(frac=1)
    labels = features["WINorLOSS"]
    features.drop("WINorLOSS", axis=1, inplace=True)
    return features, labels

# Input function used with dnn_classifier returns iterators of features, labels
def input_function(features, targets, batch_size=1, shuffle=True, num_epochs=None):

    # Construct a dataset, and configure batching/repeating.
    ds = tf.data.Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    # Return the next batch of data
    feature, label = ds.make_one_shot_iterator().get_next()
    return feature, label


if __name__ == "__main__":

    print 'Hello'

    if create_data:
        #Create necessary directory for data
        if not os.path.exists(season+"_season"):
            os.mkdir(season+"_season")

        # Creates csv for average team stats
        create_team_csv()

        # Creates csv for season keys
        create_season_csv()

        # Creates features csv
        create_raw_features_csv()

        # Fix feature values
        feature_engineering()

    features, labels = load_features_labels()
    training_testing_intersection = int(labels.size*0.6)
    testing_validation_intersection = int(labels.size*0.8)
    training_features = {"data": features[:training_testing_intersection]}
    training_labels = labels[:training_testing_intersection]
    testing_features = {"data": features[training_testing_intersection:testing_validation_intersection]}
    testing_labels = labels[training_testing_intersection:testing_validation_intersection]
    validation_features = {"data": features[testing_validation_intersection:labels.size]}
    validation_labels = labels[testing_validation_intersection: labels.size]

    # Feature column for classifier
    feature_columns = [tf.feature_column.numeric_column("data", shape=features.shape[1])]

    training_input_fn = lambda: input_function(training_features, training_labels, batch_size=200)

    # Testing input fuction, returning iterator, shuffle automatically on
    testing_input_fn = lambda: input_function(testing_features, testing_labels, batch_size=200)

    # Prediction input function, one epoch
    prediction_input_fn_training = lambda: input_function(training_features, training_labels, num_epochs=1,shuffle=False)

    # Prediction input function, one epoch
    prediction_input_fn_testing = lambda: input_function(testing_features, testing_labels, num_epochs=1, shuffle=False)

    # Prediction input validation function, one epoch
    prediction_input_fn_validation = lambda: input_function(validation_features, validation_labels, num_epochs=1, shuffle=False)

    print 'Setting up classifier'
    dnn_classifier = tf.estimator.DNNClassifier(
        # model_dir=os.getcwd() + "/model/",
        feature_columns=feature_columns,
        hidden_units=[20, 10, 20],
        # optimizer = tf.train.AdamOptimizer(
        #     learning_rate=0.005,
        # )
        optimizer = tf.train.FtrlOptimizer(
            learning_rate=0.01,
            l1_regularization_strength=0.001
        )
    )

    training_error = []
    testing_error = []

    # Loop for training
    for i in range(0, 20):
        print '------------------------'
        print 'RUN: ', i + 1
        print '------------------------'
        start_time = time.time()
        _ = dnn_classifier.train(
            input_fn=training_input_fn,
            steps=50
        )
        end_time = time.time()
        print 'Training classifier: ', end_time - start_time

        # Calculate log loss
        training_predictions = list(dnn_classifier.predict(input_fn=prediction_input_fn_training))  # Array of prediction percentages
        training_probabilities = np.array([item['probabilities'] for item in training_predictions])  # 2d array of percentages of [0.043, ...]
        training_class_ids = np.array([item['class_ids'][0] for item in training_predictions])  # Array of prediction of 7
        training_pred_one_hot = tf.keras.utils.to_categorical(training_class_ids,2)  # 2d one hot array of [0. 0. ... 1. 0. 0.]

        testing_predictions = list(dnn_classifier.predict(input_fn=prediction_input_fn_testing))
        testing_probabilities = np.array([item['probabilities'] for item in testing_predictions])
        testing_class_ids = np.array([item['class_ids'][0] for item in testing_predictions])
        testing_pred_one_hot = tf.keras.utils.to_categorical(testing_class_ids, 2)

        training_log_loss = metrics.log_loss(training_labels, training_pred_one_hot)
        testing_log_loss = metrics.log_loss(testing_labels, testing_pred_one_hot)

        training_error.append(training_log_loss)
        testing_error.append(testing_log_loss)

        print("%0.2f" % training_log_loss)
        print("%0.2f" % testing_log_loss)



    # Calculate final predictions (not probabilities, as above).
    testing_predictions = dnn_classifier.predict(input_fn=prediction_input_fn_testing)
    testing_predictions = np.array([item['class_ids'][0] for item in testing_predictions])
    testing_accuracy = metrics.accuracy_score(testing_labels, testing_predictions)
    print("Testing accuracy: %0.2f" % testing_accuracy)

    validation_predictions = dnn_classifier.predict(input_fn=prediction_input_fn_validation)
    validation_predictions = np.array([item['class_ids'][0] for item in validation_predictions])
    validation_accuracy = metrics.accuracy_score(validation_labels, validation_predictions)
    print("Validation accuracy: %0.2f" % validation_accuracy)

    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.plot(training_error, label="training")
    plt.plot(testing_error, label="testing")
    plt.legend()
    plt.show()

    # Save model
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns=feature_columns)
    dnn_classifier.export_savedmodel('./saved_model/season_'+season, export.build_parsing_serving_input_receiver_fn(feature_spec))
