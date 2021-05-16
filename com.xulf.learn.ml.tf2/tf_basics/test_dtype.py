import tensorflow as tf
import numpy as np
import unittest
from cmp_util import *
from except_util import *

class TestDtype(unittest.TestCase):
    """
        参考tf文档： https://www.tensorflow.org/api_docs/python/tf/dtypes
        # tf数据类型：
        # 数值类型：tf.intx, tf.uintx, tf.floatx, dtf.ouble (x代表位数，16， 32， 64）
        # 逻辑类型: tf.bool
        # 字符串: tf.string
        # 注:
        #   1. 这些类型都定义在tf.dtypes.xx 下， tf下的是alias
        #   2. 这些类型都是tf.dtypes.Dtype的子类
        #   3. 尽可能使用tf的数值类型：python native类型需要tf猜测， 而np.array对gpu是不兼容的
    """

    def test_Dtype(self):
        d = tf.int32
        assert d.is_integer
        assert not d.is_floating
        assert not d.is_bool
        assert_equal(2**31-1, d.max)
        assert_equal(-2**31, d.min)

    def test_cast(self):
        # 类型转换：
        #   方法签名： tf.dtypes.cast(tensor, dtype, name=None)
        #     dtype:  数值类型的tf.Dtype
        #     返回值：dtype类型的tensor

        # int => float
        t = tf.constant(3, dtype=tf.int32)
        cast_t = tf.cast(t, tf.float32)
        assert_equal(cast_t, 3)
        assert cast_t.dtype == tf.float32

        # float=>int:  小数会被截断
        t = tf.constant(3.1, dtype=tf.float32)
        cast_t = tf.cast(t, tf.int32)
        assert_equal(cast_t, 3)
        assert cast_t.dtype == tf.int32

        # 数值类型 => bool
        t = tf.constant([1, 0], dtype=tf.float32)
        cast_t = tf.cast(t, tf.bool)
        assert_equal([True, False], cast_t)

        # bool => 数值类型
        t = tf.constant([True, False], dtype=tf.bool)
        cast_t = tf.cast(t, tf.float32)
        assert_equal([1, 0], cast_t)

        # string ≠> 数值类型
        t = tf.constant('1')
        cast_lambda = lambda: tf.cast(t, tf.float32)
        assert_except(cast_lambda)

    def test_python_native_convert(self):
        # 与python native间的类型关系：
        #       python int => tf.int32
        #       python float=>tf.float32
        #       python string = tf.string
        t = tf.constant(3)
        assert t.dtype == tf.int32

        t = tf.constant(3.)
        assert t.dtype == tf.float32

        t = tf.constant('abc')
        assert t.dtype == tf.string

        #   与python容器对应:
        #       python scalar => 0-d tensor
        #       python 1-d array => 1-d tensor
        #       python 2-d array => 2-d tensor
        t = tf.zeros_like(0) #tf.Tensor(0, shape=(), dtype=int32)
        assert_equal(0, t)
        assert_shape(t, ())

        t = tf.zeros_like(['a', 'b', 'c'])
        assert_equal([b'', b'', b''], t)

        t = tf.zeros_like([[1, 2], [3, 4]])
        assert_shape(t, (2, 2))
        assert_equal(0, t)

    def test_as_dtype(self):
        # 方法签名： tf.dtypes.as_type(type_value)
        #     type_value: numpy.dtype, string, Dtype
        #     返回值：
        d = tf.as_dtype(np.int32)
        assert d == tf.int32

        d = tf.as_dtype('int32')
        assert d == tf.int32

        d = tf.as_dtype(tf.int32)
        assert d == tf.int32

