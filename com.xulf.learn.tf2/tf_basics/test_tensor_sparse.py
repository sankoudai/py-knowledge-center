import unittest
import tensorflow as tf
from cmp_util import *


class TestSparse(unittest.TestCase):
    '''
        SparseTensor:
        - 由非零indices，非零values, dense_shape组成.
        - 有特殊处理的op, 普通tensor上的op一般不可使用
        - 不可indexing
        参考：
            1. https://www.tensorflow.org/api_docs/python/tf/sparse
    '''

    def test_create(self):
        # 通过构造函数
        # tf.sparse.SparseTensor
        #   签名：tf.sparse.SparseTensor(indices, values, dense_shape)
        #     indices: [N, ...]， 其中N为非零值的个数， ...为k维的下标
        #     values: 1-d tensor， 长为N
        #     dense_shape: 长度为k
        #   说明：
        #       1.要求indices是字典序的，否则创建后应该使用tf.sparse.reorder(sparse_t)来处理下。
        #         这是因为很多算子做了这个假定
        sparse_t = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 3])
        sparse_t = tf.sparse.reorder(sparse_t)
        assert_equal_sparse(sparse_t, [[1, 0, 0],
                                       [0, 0, 2],
                                       [0, 0, 0]])


    def test_conversion(self):
        # 转化为普通Tensor
        # tf.sparse.to_dense(sparse_densor, default_value=None,..,name=None)
        sparse_t = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 3])
        t = tf.sparse.to_dense(sparse_t)
        assert_equal(t, [[1, 0, 0],
                         [0, 0, 2],
                         [0, 0, 0]])

        # 由普通Tensor转SparseTensor
        # tf.sparse.from_dense(tensor, name=None)
        t = tf.constant([[1, 0, 0],
                         [0, 0, 2],
                         [0, 0, 0]])
        sparse_t = tf.sparse.from_dense(t)
        assert_equal([[0, 0], [1, 2]], sparse_t.indices)
        assert_equal([1, 2], sparse_t.values)
        assert_equal([3, 3], sparse_t.dense_shape)
