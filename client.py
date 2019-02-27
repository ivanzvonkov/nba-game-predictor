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
        df.drop('Unnamed: 0', axis=1, inplace=True)
        feature = df.loc[ (df['h._'+home_team] == 1) & (df['a._'+away_team] == 1)]
        return feature

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
    def predict_print_out(self, one_feature):
        prediction_value, prediction_accuracy = self.predict_one(one_feature)
        print 'Guessing it\'s ' + str(prediction_value) + ' with ' + str(
            "%.2f" % prediction_accuracy) + '% accuracy.'

    def __init__(self):

        print 'Hello'

        feature = self.load_feature('ATL','BOS')

        # Feature column for classifier
        feature_columns = [tf.feature_column.numeric_column("data", shape=feature.shape[1])]

        with tf.Session() as sess:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], './saved_model/1551240267')
            self.predictor = tf.contrib.predictor.from_saved_model('./saved_model/1551240267')
            self.predict_print_out(feature)



Client()