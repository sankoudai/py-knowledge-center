import tensorflow as tf
import unittest
from cmp_util import *
from except_util import *
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class TestMatMatOp(unittest.TestCase):
    def test_matmul(self):
        # 矩阵乘法：matmul, python3可以使用a @ b
        #   方法签名: tf.linalg.matmul(a, b, transpose_a=False, transpose_b=False,
        #               adjoint_a=False, adjoint_b=False, a_is_sparse=False, b_is_sparse=False, name=None）
        #      a, b:  a[.., M, N] x b[.., N, L]， ..维度指定batch size，对应batch的矩阵做乘法
        #      其余参数： 指定相乘前，对应矩阵需要做的变换，如transpose_a=True，则结果为a转置 * b
        #      返回值:  c[.., M, L]

        # a x b
        a = tf.ones([2, 3])  # (2, 3)
        b = tf.ones([3, 2])  # (3, 2)
        t = tf.linalg.matmul(a, b)  # (2, 2)
        assert_equal([[3, 3],
                      [3, 3]], t)

        # batch a x batch b: batch维度必须对齐
        batch_a = tf.ones([3, 2, 5])
        batch_b = tf.ones([3, 5, 2])
        t = batch_a @ batch_b
        expected_t = tf.fill([3, 2, 2], 5)
        assert_equal(expected_t, t)

        # batch a x b:  b is broadcast (to batches of a)
        batch_a = tf.ones([3, 2, 5])
        b = tf.ones([5, 2])
        t = batch_a @ b
        expected_t = tf.fill([3, 2, 2], 5)
        assert_equal(expected_t, t)

        # a x batch b: a is broadcast (to batches of b)
        a = tf.ones([2, 5])
        batch_b = tf.ones([3, 5, 2])
        t = a @ batch_b
        expected_t = tf.fill([3, 2, 2], 5)
        assert_equal(expected_t, t)

    def test_solve(self):
        # 解线性方程组
        #   方法签名: tf.linalg.solve(m, rhs, adjoint=False, name=None)
        #      m, rhs:  m[.., M, M]系数矩阵， rhs[.., M, K]方程组右侧结果; ..维度指定batch，解对应batch的方程组
        #      返回值:  c[.., M, K]

        # m可逆： 得到唯一解
        m = tf.constant([[1, 2],
                         [3, 4]], dtype=tf.float32) #(2, 2)
        expected_x = tf.constant([[3],
                                  [4]], dtype=tf.float32) #(2, 1)
        rhs = m @ expected_x

        x = tf.linalg.solve(m, rhs)
        assert_equal(expected_x, x, tol=1e-6)

        #m不可逆: 报错
        m = tf.constant([[1, 1],
                         [1, 1]], dtype=tf.float32)
        expected_x = tf.constant([[3],
                                  [4]], dtype=tf.float32)
        rhs = m @ expected_x

        f = lambda: tf.linalg.solve(m, rhs)
        assert_except(f)
