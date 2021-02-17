import tensorflow as tf
from cmp_util import *
from unittest import TestCase


class TensorConstructTest(TestCase):
    def test_constant(self):
        c1 = tf.constant(1, dtype=tf.int32)
        assert_val(c1, 1)

        val = [[3, 1], [2, 4]]
        arr_tensor = tf.constant(val, dtype=tf.int32)
        assert_val(arr_tensor, val)

    def test_single_val_tensor(self):
        # special values: tf.zeros, tf.ones
        t = tf.zeros([2, 3], dtype=tf.int32)
        assert_val(t, [[0, 0, 0], [0, 0, 0]])

        refer_tensor = tf.constant([[1, 2], [2, 2]])
        t = tf.zeros_like(refer_tensor, dtype=tf.int32)
        assert_val(t, [[0, 0], [0, 0]])

        # specified value: tf.fill(dims, scalar, name)
        shape = [2, 3]
        t = tf.fill(shape, 3)
        assert_val(t, [[3, 3, 3], [3, 3, 3]])

    def test_seq(self):
        # tf.linspace(start, stop, num)
        # tf1中有别名tf.lin_space
        t = tf.linspace(0., 3.0, 4)
        assert_val(t, [0., 1., 2., 3.])

        # tf.range(start, limit=None, delta=1): limit not included!!
        t = tf.range(0.0, 3.0, 1.0)
        assert_val(t, [0.0, 1.0, 2.0])
        print(t)






