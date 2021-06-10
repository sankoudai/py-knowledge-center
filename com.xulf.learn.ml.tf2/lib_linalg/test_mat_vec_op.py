import tensorflow as tf
import unittest
from cmp_util import *
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # 为了兼容mac上的bug

class TestMatVecOp(unittest.TestCase):
    def test_matvec(self):
        # 矩阵乘以向量
        #   方法签名：tf.linalg.matvec(a, b, transpose_a=False, adjoint_a=False,
        #               a_is_sparse=False, b_is_sparse=False, name=None)
        #      a, b:  a[.., M, N] x b[.., N], ..维度指定batch size，对应batch的矩阵乘向量
        #      其余参数： 指定相乘前，对应矩阵需要做的变换，如transpose_a=True，则结果为a转置 * b
        #      返回值:  c[.., M]

        # m x v
        m = tf.ones([2, 3])
        v = tf.ones([3])
        t = tf.linalg.matvec(m, v)
        assert_equal([3, 3], t)

        # batch_m x batch v:  batch dim must be of same shape
        batch_m = tf.ones([5, 4, 2, 3])
        batch_v = tf.ones([5, 4, 3])
        t = tf.linalg.matvec(batch_m, batch_v)
        expected_t = tf.fill([5, 4, 2], 3)
        assert_equal(expected_t, t)

        # batch_m x v:  v is broadcast (to batches of m)
        batch_m = tf.ones([5, 4, 2, 3])
        v = tf.ones([3])
        t = tf.linalg.matvec(batch_m, v)
        expected_t = tf.fill([5, 4, 2], 3)
        assert_equal(expected_t, t)

        # m x batch_v:  m is broadcast (to batches of v)
        m = tf.ones([2, 3])
        batch_v = tf.ones([5, 4, 3])
        t = tf.linalg.matvec(m, batch_v)
        expected_t = tf.fill([5, 4, 2], 3)
        assert_equal(expected_t, t)