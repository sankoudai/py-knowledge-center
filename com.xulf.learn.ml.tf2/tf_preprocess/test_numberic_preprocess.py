import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import unittest
from tensorflow.keras.layers.experimental import preprocessing

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)


class TestNumberic(unittest.TestCase):
    def setUp(self) -> None:
        # 读入内存
        headers = ["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
                   "Viscera weight", "Shell weight", "Age"]
        abalone_train = pd.read_csv('data/abalone_train.csv', sep=',', names=headers)

        # 拆分features, label
        features = abalone_train.copy()
        labels = features.pop('Age')

        # 转化为numpy array:
        features = np.array(features)

        self.features, self.labels = features, labels


    def test_normalize(self):
        normalize_layer = preprocessing.Normalization()
        normalize_layer.adapt(self.features)

        model = tf.keras.Sequential([
            normalize_layer,
            layers.Dense(10),
            layers.Dense(1)
        ])
        model.compile(loss=tf.losses.MeanSquaredError(),
                      optimizer=tf.optimizers.Adam())
        model.fit(self.features, self.labels, validation_split=0.2, epochs=2)
