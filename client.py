import tensorflow as tf
import pandas as pd


class Client:

    # Input two teams, output which one won
    # Create feature from two teams
    # Feed into classifier
    # Use result in determining playoff bracket

    def load_feature(self, home_team, away_team):
        df = pd.read_csv("features.csv")
        df.drop("WINorLOSS", axis=1, inplace=True)
        #df.drop('Unnamed: 0', axis=1, inplace=True)
        feature = df.loc[ (df['h._'+home_team] == 1) & (df['a._'+away_team] == 1)]
        return feature.values

    # Predict one 2d array feature
    def predict_one(self, one_feature):

        model_input = tf.train.Example(features=tf.train.Features(feature={
            'data': tf.train.Feature(float_list=tf.train.FloatList(value=one_feature))
        }))

        model_input = model_input.SerializeToString()

        output_dict = self.predictor({u'inputs': [model_input]})
        prediction_list = list(output_dict[u'scores'][0])
        prediction_value = prediction_list.index(max(prediction_list))
        prediction_accuracy = max(prediction_list) * 100

        return prediction_value, prediction_accuracy

    # Predict random digit from data
    def predict_value(self, one_feature):
        prediction_value, prediction_accuracy = self.predict_one(one_feature)
        if prediction_value == 1:
            winning_team =   self.home_team
            losing_team = self.away_team
        else:
            winning_team = self.away_team
            losing_team = self.home_team

        print 'Guessing ' + winning_team + ' beat '+ losing_team + ' '+ str(
            "%.2f" % prediction_accuracy) + '% accuracy.'

        return winning_team

    def __init__(self):

        current_round_west = ['GSW', 'HOU', 'LAC', 'POR', 'MEM', 'SAS', 'DAL', 'NOP']
        current_round_east = ['ATL', 'CLE', 'CHI', 'TOR', 'WAS', 'MIL', 'BOS', 'BRK']
        next_round_west = []
        next_round_east = []

        west_finalist_chosen = False
        east_finalist_chosen = False

        with tf.Session() as sess:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], './saved_model/1551240267')
            self.predictor = tf.contrib.predictor.from_saved_model('./saved_model/1551240267')

            while True:

                # More than one team left in west current round
                if not west_finalist_chosen and len(current_round_west) > 1:
                    self.home_team = current_round_west.pop(0)
                    self.away_team = current_round_west.pop(len(current_round_west)-1)

                # More than one team left in east current round
                elif not east_finalist_chosen and len(current_round_east) > 1:
                    self.home_team = current_round_east.pop(0)
                    self.away_team = current_round_east.pop(len(current_round_east)-1)

                elif east_finalist_chosen and west_finalist_chosen:
                    self.home_team = current_round_west.pop(0)
                    self.away_team = current_round_east.pop(0)

                # Predicting
                feature = self.load_feature(self.home_team, self.away_team)
                winning_team = self.predict_value(feature[0])

                if east_finalist_chosen and west_finalist_chosen:
                    print winning_team + ' has won an NBA Championship!'
                    break

                # West teams just played, add west team to next round
                elif not west_finalist_chosen and len(current_round_west) >= 0:
                    print winning_team + ' moving on'
                    next_round_west.append(winning_team)
                    # Round over
                    if len(current_round_west) == 0:
                        current_round_west = next_round_west
                        next_round_west = []
                        print 'West Round Over, teams that move on '+ str(current_round_west) + '\n'
                        if len(current_round_west) == 1:
                            west_finalist_chosen = True
                            print 'The WCF is ' + current_round_west[0] + '!\n'

                # East teams just played, add west team to next round
                elif not east_finalist_chosen and len(current_round_east) >= 0:
                    print winning_team + ' moving on'
                    next_round_east.append(winning_team)
                    # Round over
                    if len(current_round_east) == 0:
                        current_round_east = next_round_east
                        next_round_east = []
                        print 'East Round Over, teams that move on ' + str(current_round_east) + '\n'
                        if len(current_round_east) == 1:
                            east_finalist_chosen = True
                            print 'The ECF is ' + current_round_west[0] + '!\n'







Client()