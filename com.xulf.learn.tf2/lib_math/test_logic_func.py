import tensorflow as tf
from tensorflow import math as tfm
import unittest
from cmp_util import *

class TestLogic(unittest.TestCase):
    def test_logic(self):
        # and: tf.math.logical_and(x, y, name=None)
        # or: tf.math.logical_or(x, y, name=None)
        # not: tf.math.logical_not
        t1 = tf.constant([False, True,  False])
        t2 = tf.constant([True, True, False])
        t = tfm.logical_and(t1, t2)
        assert_equal([False, True, False], t)

        t1 = tf.constant([False, True,  False])
        t2 = tf.constant([True])
        t = tfm.logical_and(t1, t2)
        assert_equal(t1, t)