import tensorflow as tf
from tensorflow import math as tfm
import numpy as np
import unittest
from cmp_util import *

class TestCmp(unittest.TestCase):
    def setUp(self):
        self.t_dim1 = tf.constant([1, 2, 3, 2])
        self.t_dim2 = tf.constant([
            [1, 3, 4, 1],
            [1, 4, 3, 1],
            [4, 3, 2, 1]
        ])

        self.t_dim3 = [
            [[0, 0, 2],
             [0, 1, 4],
             [0, 2, 2]],

            [[1, 0, 4],
             [1, 1, 3],
             [1, 2, 2]]
        ]

    def test_argmax_min(self):
        # argmax: 获取沿某维度取值最大的下标
        # tf.math.argmax(input, axis=None, output_type=tf.dtypes.int64, name=None), axis默认沿第0维
        # 存在别名: tf.argmax
        index_max = tfm.argmax(self.t_dim1)
        assert_equal(2, index_max)

        index_max = tfm.argmax(self.t_dim2, axis=0)
        assert_equal([2, 1, 0, 0], index_max)

        index_max = tfm.argmax(self.t_dim3, axis=1)
        assert_equal([[0, 2, 1],
                      [0, 2, 0]], index_max)

        # argmin: 用法同argmin， 获取沿某个维度取值最大的小标

    def test_top(self):
        # in_top_k: 用于分类问题
        # 方法签名：tf.math.in_top_k(targets, preds, k, name=None)
        #   targets： batch_size 的1d tensor (int32 or int64)
        #   preds：   batch_size x num_class 的2d tensor (float32)
        #   返回值：   batch_size 维度的1d bool tensor
        #            out[i] = True 当且仅当preds[i][targets[i]]在preds[i]的topK之中
        targets = tf.constant([1, 2, 2, 0])
        preds = tf.constant([
            [0.1, 0.15, 0.5, 0.25],
            [0.1, 0.3, 0.4, 0.2],
            [0.1, 0.1, 0.5, 0.3],
            [0.1, 0.2, 0.5, 0.2]
        ])

        in_top_k = tfm.in_top_k(targets, preds, 1)
        assert_equal([False, True, True, False], in_top_k)

        in_top_k = tfm.in_top_k(targets, preds, 3)
        assert_equal([True, True, True, False], in_top_k)

        # top_k： 获取沿最后维度取的topK个值与其下标
        # 方法签名： tf.math.top_k(tensor, k=1, sorted=True, name=None)
        #     sorted:  如果True, 返回的K个值按降序排列
        #     返回值: values - 沿最后维度的topK个值
        #            indices - 对应下标
        pred_1d = tf.constant([0.1, 0.2, 0.5, 0.2])
        top_k, top_k_indices = tfm.top_k(pred_1d, k=2)
        assert_equal([0.5, 0.2], top_k,  tol=1e-6)
        assert_equal([2, 1], top_k_indices)

        preds_2d = tf.constant([
            [0.1, 0.15, 0.5, 0.25],
            [0.1, 0.3, 0.4, 0.2],
            [0.1, 0.1, 0.5, 0.3]
        ])
        top_k, top_k_indices = tfm.top_k(preds_2d, k=2)
        assert_equal([[0.5, 0.25],
                      [0.4, 0.3],
                      [0.5, 0.3]
                     ],
                      top_k,  tol=1e-6)
        assert_equal([[2, 3],
                      [2, 1],
                      [2, 3]],
                      top_k_indices)

    def test_cmp(self):
        # 大于: tf.math.greater(x, y, name=None) -> bool类型的tensor
        # 大于等于: tf.math.greater_equal
        # 小于：tf.math.less
        # 小于等于：tf.math.less_equal
        # 等于: tf.math.equal
        # 不等于： tf.math.not_equal
        t1 = tf.constant([1, 2, 3])
        t2 = tf.constant([3, 2, 1])
        t = tfm.greater(t1, t2)
        assert_equal([False, False, True], t)

        t1 = tf.constant([1, 2, 3])
        t2 = tf.constant([2])
        t = tfm.greater(t1, t2)
        assert_equal([False, False, True], t)

    def test_maximum_minimum(self):
        # 取最大: tf.math.maximum(x, y, name=None)
        # 取最小: tf.math.minimum
        t1 = tf.constant([1, 2, 3])
        t2 = tf.constant([3, 2, 1])
        t = tfm.maximum(t1, t2)
        assert_equal([3, 2, 3], t)

        t1 = tf.constant([1, 2, 3])
        t2 = tf.constant([2])
        t = tfm.maximum(t1, t2)
        assert_equal([2, 2, 3], t)
