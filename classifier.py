import time
import tensorflow as tf
import numpy as np
import gzip
import matplotlib.pyplot as plt
from tensorflow.python.data import Dataset
from sklearn import metrics
from tensorflow.python.estimator.export import export


# Loads features from file and returns features dict
def load_data():
    filename = 'nba-games-stats-from-2014-to-2018.csv'
    return




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
    load_data()
