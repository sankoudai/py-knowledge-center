import tensorflow as tf
import numpy as np
import unittest
np.set_printoptions(precision=4)

class TestDatasetUsage(unittest.TestCase):
    def setUp(self) -> None:
        f = tf.random.uniform([1000, 2], minval=0, maxval=1.0)
        noise = tf.random.uniform([1000], minval=0, maxval=0.05)

        self.features = tf.data.Dataset.from_tensor_slices(f)
        self.labels = tf.data.Dataset.from_tensor_slices(f[:,0] + f[:, 1] + noise)
        self.dataset = tf.data.Dataset.zip((self.features, self.labels))

    def test_iterate(self):
        ds = tf.data.Dataset.from_tensor_slices([1, 2, 3])
        for ele in ds:
            print(ele)

    def test_model_fit(self):
        # before fit: shuffle and batch
        ds = self.dataset.shuffle(1000).batch(10)

        # fit
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1)
        ])
        model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.SGD())
        model.fit(ds, epochs=10)

        # predict
        pred = model.predict([[1, 2]])
        print(pred)






