import tensorflow as tf
from tensorflow import math as tfm
import unittest
from cmp_util import *

class TestReduce(unittest.TestCase):
    def test_bool_reduce(self):
        # reduce_all: 等价于np.all
        #   方法签名：tf.math.reduce_all(tensor, axis=None,  keepdims=False, name=None)
        #       tensor: bool type
        #       axis: None or int or int list, 沿该轴做聚合。 如果是None，沿所有轴聚合， 返回scalar 的Bool值
        #       keepdims: 如果True, 保留聚合的轴(维度大小为1）； 否则返回结果维度-1
        #       返回值： bool tensor

        t = tf.constant([
            [True, True, False],
            [True, True, False],
            [True, False, True]
        ])
        reduced_t = tfm.reduce_all(t)
        assert_equal(False, reduced_t)

        reduced_t = tfm.reduce_all(t, axis=0)
        assert_equal([True, False, False], reduced_t)

        reduced_t = tfm.reduce_all(t, axis=1)
        assert_equal([False, False, False], reduced_t)

        t = tf.constant([
            [[True, True, False],
            [True, False, True]],

            [[True, True, True],
            [True, True, True]]
        ])
        reduced_t = tfm.reduce_all(t, axis=[1, 2])
        assert_equal([False, True], reduced_t)

        # reduce_any: 等价于np.any， 用法同reduce_all

    def test_numeric_reduce(self):
        # reduce_min: 取最小，tf.math.reduce_min(tensor, axis=None,  keepdims=False, name=None)
        # reduce_max: 取最大, tf.math.reduce_max(tensor, axis=None, keepdims=False, name=None)
        # reduce_sum: 求和， tf.math.reduce_sum(tensor, axis=None, keepdims=False, name=None)
        # reduce_prod: 求乘积，tf.math.reduce_prod(tensor, axis=None, keepdims=False, name=None)
        # reduce_mean: 取平均, tf.math.reduce_mean(tensor, axis=None, keepdims=False, name=None)
        # reduce_std: 取标准差，tf.math.reduce_std(tensor, axis=None, keepdims=False, name=None)
        # reduce_variance: 取方差tf.math.reduce_variance(tensor, axis=None, keepdims=False, name=None)
        t = tf.constant([
            [[1, 2, 3],
             [3, 4, 5]],
            [[2, 3, 4],
             [1, 1, 1]]
        ])

        reduced_t = tfm.reduce_sum(t, axis=1)
        assert_equal(
            [[4, 6, 8],
            [3, 4, 5]],
            reduced_t)