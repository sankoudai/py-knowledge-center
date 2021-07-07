import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import unittest

class TestCsv(unittest.TestCase):
    def test_load_numberics_to_numpy(self):
        '''文件特点：所有列均数值类型，可加载到内存'''

        # 读入内存
        headers=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
                "Viscera weight", "Shell weight", "Age"]
        abalone_train = pd.read_csv('data/abalone_train.csv', sep=',', names=headers)

        # 拆分features, label
        features = abalone_train.copy()
        labels = features.pop('Age')

        #转化为numpy array:
        features = np.array(features)

        #模型训练
        model = tf.keras.Sequential([
            layers.Dense(32),
            layers.Dense(1)
        ])
        model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam())
        model.fit(features, labels, validation_split=0.2, epochs=2)


