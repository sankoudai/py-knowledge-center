import unittest
import tensorflow as tf
from cmp_util import *


class TestRagged(unittest.TestCase):
    '''
        RaggedTensor: 普通Tensor是方正的 mxn，或者各axis等长的； 而不等长的用RaggedTensor来表示
        参考：
            1. https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/ragged
            2. https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/RaggedTensor
    '''

    def test_create(self):
        # tf.ragged.constant(pylist, dtype=None, .., name=None)
        ragged_t = tf.ragged.constant([[1],
                                       [2, 3],
                                       [4, 5]])

        # 一些其他的op也可以创建RaggedTensor, 如tf.strings.split

    def test_conversion(self):
        # to and from tf.Tensor
        ragged_t = tf.ragged.constant([[1],
                                       [2, 3],
                                       [4, 5]])
        t = ragged_t.to_tensor(default_value=100)
        assert_equal([[1, 100],
                      [2, 3],
                      [4, 5]], t)

        t = tf.constant([[1, 0],
                         [2, 3],
                         [4, 5]])
        ragged_t = tf.RaggedTensor.from_tensor(t, padding=0)
        assert_equal(ragged_t[0], [1])
        assert_equal(ragged_t[1], [2, 3])
        assert_equal(ragged_t[2], [4, 5])

        # to and from tf.SparseTensor
        ragged_t = tf.ragged.constant([[1],
                                       [4, 1]])
        spart_t = tf.RaggedTensor.to_sparse(ragged_t)
        assert_equal([1, 4, 1], spart_t.values)
        assert_equal([[0,0],[1,0], [1, 1]], spart_t.indices)
        assert_equal((2, 2), spart_t.shape)

