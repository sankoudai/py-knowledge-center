import unittest
import tensorflow as tf
from cmp_util import *

class TestShape(unittest.TestCase):
    def test_reshape(self):
        # reshape:
        #   方法签名： tf.reshape(tensor, shape)
        #     返回值：新tensor， 但与原tensor共享内存(tf中tensor不可变）
        #   说明：将相邻的维度拆分或合并

        t = tf.constant([[0, 1, 2, 3, 4, 5],
                         [6, 7, 8, 9, 10, 11]]) #（2， 6）
        # 将1维度拆分: 6=2x3
        splitdim_t = tf.reshape(t, [2, 2, 3])
        assert_equal([[[0, 1, 2],
                       [3, 4, 5]],
                      [[6, 7, 8],
                       [9, 10, 11]]], splitdim_t)

        # 合并0维, 1维: 2 x 6 = 12
        mergedim_t = tf.reshape(t, [12])
        print(mergedim_t)
        assert_equal([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], mergedim_t)

    def test_expand_dim(self):
        ''' 增加一维 '''
        # tf.expand_dims
        t = tf.zeros((2, 3))
        assert_shape(tf.expand_dims(t, 0), (1, 2, 3))
        assert_shape(tf.expand_dims(t, 1), (2, 1, 3))
        assert_shape(tf.expand_dims(t,-1), (2, 3, 1))

        # tf.newaxis:等价写法
        new_t = t[tf.newaxis, ...]
        assert_shape(new_t, (1, 2, 3))

        new_t = t[:, tf.newaxis, :]
        assert_shape(new_t, (2, 1, 3))