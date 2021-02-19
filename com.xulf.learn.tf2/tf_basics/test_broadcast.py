import tensorflow as tf
import unittest
from cmp_util import *
from except_util import *


class TestBroadcast(unittest.TestCase):
    '''
        broadcast过程: 以t1 op t2为例说明， 其中op是element-wise的
        1. 对齐rank: 将rank较小的tensor 左侧扩充大小为1的维度
            举例：
                t1 = tf.constant([[1, 2], [3,4]]) #(2, 2)
                t2 = tf.constant([5, 6])  #(2,)  ==> [[5,6]] #(1, 2)
        2. 判断是否compatible: 对应维度大小相同，或为1
                t1.shape[i]==t2.shape[i] or t1.shape[i]=1 or t2.shape[i]=1
            举例： (2, 2) 与 (2, 1)是兼容的， 但(2, 3) 与(2, 2)不兼容
        3. 为1的维度复制扩充:
            举例:
                t1 = tf.constant([[1, 2], [3,4]]) #(2, 2)
                t2 = tf.constant([[5,6]])  #(1, 2) ==> [[5, 6], [5, 6]]
        参考：
            https://www.tensorflow.org/api_docs/python/tf/broadcast_to
            https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html
    '''

    def test_broadcast(self):
        # rank相同
        t1 = tf.constant([[1, 2, 3]])  # (1, 3)
        t2 = tf.constant([[4],
                          [5],
                          [6]])  # (3, 1)
        t = t1 * t2
        assert_equal([[1 * 4, 2 * 4, 3 * 4],
                      [1 * 5, 2 * 5, 3 * 5],
                      [1 * 6, 2 * 6, 3 * 6]], t)

        # rank不同
        t1 = tf.constant([[1, 2, 3]])  # (1, 3)
        t2 = tf.constant([1, 2, 3])  # (3, )
        t = t1 * t2
        assert_equal([[1 * 1, 2 * 2, 3 * 3]], t)

    def test_usecase(self):
        # centering
        t = tf.constant([[1, 2],
                         [3, 6],
                         [2, 4]], dtype=tf.float32)
        mean = tf.math.reduce_mean(t, axis=0)
        assert_equal([[-1, -2],
                      [1, 2],
                      [0, 0]], t - mean)
