import tensorflow as tf
import numpy as np
import unittest
from cmp_util import *
from except_util import *


class TestIndexing(unittest.TestCase):
    def test_single_axis(self):
        ## 抽取固定间隔的行
        # t[i]：规则与python一致
        # a. 第一个元素 0
        # b. 最后一个元素 -1
        # c. slicing:  start:stop:step, stop不包含

        # 1d
        t = tf.range(10)
        assert_equal(0, t[0])
        assert_equal(9, t[-1])
        assert_equal([0, 3, 6], t[0:9:3])
        assert_equal([0, 3, 6, 9], t[::3])
        assert_equal([9, 6, 3, 0], t[-1::-3])

        # 2d: 作用在第0维度上
        t = tf.constant([[0, 1],
                         [2, 3]])
        assert_equal([0, 1], t[0])

        # 抽取指定行（按照指定顺序）
        #  gather
        #    方法签名：tf.gather(tensor, indices, axis=None, ..., name=None)
        #       indices: 0d or 1-d tensor
        #       axis: int, 沿该维度抽取
        t = tf.constant([[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [10, 20, 30, 40],
                         [50, 60, 70, 80]]) # (2, 2, 4)

        # 0d-indices: 等价于t[i] extract specific row
        subrows_t = tf.gather(t, indices=1, axis=0)
        assert_equal([5, 6, 7, 8], subrows_t)

        # 1d-indices: [t[i], t[j],.., t[k]], extract and reorder rows
        subrows_t = tf.gather(t, indices=[3,1], axis=0)
        assert_equal([[50, 60, 70, 80],
                      [5, 6, 7, 8]], subrows_t)

        # axis is other dim
        subcol_t = tf.gather(t, indices=2, axis=1)
        assert_equal([3, 7, 30, 70], subcol_t)

    def test_multi_axis(self):
        # （多维度）固定间隔：
        #   t[i, j, k]:  各维度独立， 都适用单维度规则
        t = tf.constant([[[1, 2, 3, 4],
                          [5, 6, 7, 8]],
                         [[10, 20, 30, 40],
                          [50, 60, 70, 80]]]) # (2, 2, 4)
        assert_equal(80, t[1, 1, 3])
        assert_equal([10, 30], t[1, 0, ::2])
        assert_equal([[1, 5], [10, 50]], t[:, :, 0])

        # t[i][j][k]:  i, j, k都是数字， 等价于t[i, j, k]
        assert_equal(80, t[1][1][3])

        # （多维度）指定子tensor并重新排布：
        #   gather_nd:
        #       方法签名： gather_nd(tensor, indices, ..., name=None)
        #          indices: rank-k tensor， 前k-1维指定子tensor位置(排布），最内的一维是指定子tensor (t[i, j, k])

        t = tf.constant([[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [10, 20, 30, 40],
                         [50, 60, 70, 80]])
        # 1d-indices: 等价于t[i, j, k]
        subtensors_block = tf.gather_nd(t, indices=[1, 0])
        assert_equal(subtensors_block, t[1, 0])

        # 2d-indices:
        subtensors_block = tf.gather_nd(t, indices=[[0, 1], [2, 1]])
        assert_equal([2, 20], subtensors_block)

        subtensors_block = tf.gather_nd(t, indices=[[0], [2]])
        assert_equal([[1, 2, 3, 4],
                      [10, 20, 30, 40]], subtensors_block)