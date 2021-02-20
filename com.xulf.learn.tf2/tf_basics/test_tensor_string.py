import unittest
import tensorflow as tf
from tensorflow import strings as tfs
from cmp_util import *


class TestStringTensor(unittest.TestCase):
    def test_create(self):
        # 创建方式与numberic类似:
        scalar_t = tf.constant("abc", dtype=tf.string)
        assert_shape(scalar_t, ())
        assert_equal(b'abc', scalar_t)

        tensor_t = tf.constant(["abc", "foo", 'boo'], dtype=tf.string)
        assert_shape(tensor_t, (3,))

    def test_split(self):
        # split:
        #   方法签名：tf.strings.split(input, sep=None, maxsplit=-1, name=None)
        #   返回值： rank+1的RaggedTensor
        #
        #   注：
        #     1. tf1还有个result_type参数: 'SparseTensor' or 'RaggedTensor'，确定返回值类型
        #     2. RaggedTensor有转化为SparseTensor的方法
        scalar_t = tf.constant("foo,boo,too", dtype=tf.string)
        t = tf.strings.split(scalar_t, ',')
        print(t)
        assert_shape(t, (3,))
        assert_equal([b'foo', b'boo', b'too'], t)

        tensor_t = tf.constant(["a,b,c", "foo,boo"], dtype=tf.string)
        t = tf.strings.split(tensor_t, ',')
        assert isinstance(t, tf.RaggedTensor)
        print(t.indices)

    def test_to_number(self):
        # 数字内容的string转数值类型:
        # tf.strings.to_number
        #    方法签名： tf.strings.to_number(tensor, out_type=tf.types.float32, name=None)
        #       tensor： 每一个元素，必须是数字!
        #  (tf1: tf.string_to_number)

        t = tf.constant(['1.1', '1.2', '3'], dtype=tf.string)
        numeric_t = tf.strings.to_number(t)
        assert_equal([1.1, 1.2, 3], numeric_t, tol=1e-6)

        # 取hash:
        # tf.strings.to_hash_bucket_strong
        #   方法签名：tf.strings.to_hash_bucket_strong(tensor, num_buckets, key, name=None)
        #      tensor: 类型为string的tensor
        #      num_buckets: 桶数
        #      key: [int1, int2]类型的hash key
        #     返回值：int64类型的tensor，取值范围[0, num_buckets)
        t = tf.constant(['foo', 'xxar'])
        t = tf.strings.to_hash_bucket_strong(t, num_buckets=3, key=[100, 300])
        assert_equal([1, 0], t)

    def test_usecase(self):
        # seq feature处理：string->int64 tensor
        raw_tensor = tf.constant(['36,42, 53,79', '1,2, 3,4'])
        num_ragged_tensor = tf.strings.to_number(tf.strings.strip(tf.strings.split(raw_tensor, ',')))
        num_tensor = num_ragged_tensor.to_tensor()
        assert_equal([[36, 42, 53, 79],
                      [1, 2, 3, 4]], num_tensor)
        assert isinstance(num_tensor, tf.Tensor)

        # string feature: hash
        raw_tensor = tf.constant(['abc', 'def'])
        hash_tensor = tf.strings.to_hash_bucket_strong(raw_tensor, num_buckets=10000, key=[100, 300])
        assert_equal([7757, 1822], hash_tensor)