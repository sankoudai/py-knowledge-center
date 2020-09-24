import unittest
import tensorflow as tf
from cmp_util import *

class ShapeOpTest(unittest.TestCase):
    def test_reshape(self):
        t = tf.zeros((2, 3,  3))
        assert_shape(tf.reshape(t, [2, 9]), [2, 9])
        assert_shape(tf.reshape(t, [-1, 1]), (18, 1))

    def test_expand_dim(self):
        ''' 增加一维 '''

        t = tf.zeros((2, 3))
        assert_shape(tf.expand_dims(t, 0), (1, 2, 3))
        assert_shape(tf.expand_dims(t, 1), (2, 1, 3))
        assert_shape(tf.expand_dims(t,-1), (2, 3, 1))
