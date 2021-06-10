import tensorflow as tf
from unittest import TestCase
from cmp_util import *


class TestArithmetic(TestCase):
    def test_arithmetic_op(self):
        ## Element-wise operation

        t1 = tf.constant([1, 1])
        t2 = tf.constant([1, 2])
        t3 = tf.constant([1, 3])

        # 加：tf.add(a, b, name=None)
        # tf2 放到了tf.math.add
        t = tf.add(t1, t2)
        assert_equal(t, [2, 3])
        t = tf.add_n([t1, t2, t3])
        assert_equal(t, tf.add(t1, tf.add(t2, t3)))

        #减：tf.substract
        t = tf.subtract(t1, t2)
        assert_equal(t, [0, -1])

        # 乘：
        t = tf.multiply(t1, t2)
        assert_equal(t, [1, 2])

        # 除:
        # 注：tf1中 int除法返回int, tf2中返回float
        t = tf.divide(t1, t2)
        assert_equal(t, [1, 0.5])

        # 取负数
        t = tf.negative(t2)
        assert_equal(t, [-1, -2])

        # 取倒数: 入参必须是float
        # tf1 中是tf.reciprocal
        t = tf.math.reciprocal(tf.constant([1.0, 2.0], dtype=tf.float32))
        assert_equal(t, [1, 0.5])

        #取模
        # tf.math.mod(x, y, name=None)
        #   返回：x/y的余数
        x = tf.constant([4, 5])

        y = tf.constant(3)
        assert_equal([1, 2], tf.math.mod(x, y))
        y=tf.constant([2, 3])
        assert_equal([0, 2], tf.math.mod(x, y))