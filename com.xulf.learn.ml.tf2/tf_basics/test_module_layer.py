import tensorflow as tf
import tensorflow.compat.v1 as tf1
from cmp_util import *
from except_util import *
import unittest
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 为了兼容mac上的bug


class TestLayer(unittest.TestCase):
    def test(self):
        class SimpleDense(tf.keras.layers.Layer):
          def __init__(self, units=32):
              super(SimpleDense, self).__init__()
              self.units = units

          def build(self, input_shape):
              self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                       initializer='random_normal',
                                       trainable=True)
              self.b = self.add_weight(shape=(self.units,),
                                       initializer='random_normal',
                                       trainable=True)

          def call(self, inputs):
              return tf.matmul(inputs, self.w) + self.b

        dense_layer = SimpleDense()
        dense_layer(tf.constant([[2, 3.]]))
        print(dense_layer.trainable_weights)
        print(dense_layer.trainable_variables)