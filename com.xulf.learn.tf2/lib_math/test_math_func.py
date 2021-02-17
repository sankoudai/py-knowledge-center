import tensorflow as tf
from tensorflow import math as tfm

import unittest
from cmp_util import *
import math

class TestFunc(unittest.TestCase):
    def test_pow_log(self):
        # exp vs log
        t = tfm.exp(1.0)
        assert_equal(t, math.e, tol=1e-6)

        t2 = tfm.log(t)
        assert_equal(t2, 1.0, tol=1e-6)

        # pow vs ? (tf 只有自然对数， 其他底的对数需要自定义）
        t = tfm.pow(3., 2.0)
        assert_equal(t, 9., tol=1e-6)

        # special power
        t = tfm.sqrt(4.0)
        assert_equal(t, 2., tol=1e-6)

    def test_trigonometric(self):
        # 三角函数均以弧度输入输出
        assert_equal(0.0, tfm.sin(0.), tol=1e-6)
        assert_equal(1.0, tfm.cos(0.), tol=1e-6)

        #反三角函数
        assert_equal(0.0, tfm.asin(0.0), tol=1e-6)
        assert_equal(math.pi/2, tfm.acos(0.0), tol=1e-6)

        # 双曲三角函数
        # tanh：双曲正切函数
        #   说明： 奇对称函数，取值范围[-1, -1]， 在[-0.3, 0.3]与y=x非常接近， x=5.0之后变化不大
        assert_equal(0., tfm.tanh(0.), tol=1e-6)
        assert_equal(0.291, tfm.tanh(0.3), tol=0.001)
        assert_equal(0.761, tfm.tanh(1.0), tol=0.001)
        assert_equal(1.000, tfm.tanh(5.0), tol=0.001)

        assert_equal(tfm.tanh(-1.0), -1.0 * tfm.tanh(1.0), tol=1e-6)
