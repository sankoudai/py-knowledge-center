import tensorflow as tf
import unittest
from cmp_util import *
from except_util import *


class TestTensorArray(unittest.TestCase):
    '''
        tf.TensorArray: class wrap dynamic-sized tensor arrays
    '''

    def test_create(self):
        # tf.TensorArray(dtype, size=None, dynamic_size=False,  .., name=None)
        #   size: initial size
        #   dynamic_size: if True, TensorArray can grow past initial size

        # fixed-sized TensorArray： 未write的位置默认为0
        arr = tf.TensorArray(tf.float32, size=3)
        arr.write(0, tf.constant(1, dtype=tf.float32))
        arr.write(1, tf.constant(2, dtype=tf.float32))
        # arr.write(2, tf.constant(3))
        assert_equal([1, 2, 0], arr.stack())

        # dynamic sized TensorArray
        v = tf.Variable(0)

        @tf.function
        def f(x):
            ta = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
            for i in tf.range(x):
                v.assign_add(1)
                ta = ta.write(i, v)
            return ta.stack()

        t = f(5)
        assert_equal([1, 2, 3, 4, 5], t)

    def test_read_write(self):
        ta = tf.TensorArray(tf.int32, size=3)

        ta.write(1, tf.constant(1))
        assert_equal([0, 1, 0], ta.stack())

        assert_equal(1, ta.read(1))

    def test_stack_unstack(self):
        # TensorArray <-> Tensor: 纵向变换

        ta1 = tf.TensorArray(tf.int32, size=2)
        ta1.write(0, tf.constant([1, 2]))
        ta1.write(1, tf.constant([3, 4]))

        # stack(name=None): 纵向堆叠， TensorArray->Tensor, rank增1。
        #   在最前边增加一维， 将TensorArray中的每个tensor作为一行
        t = ta1.stack()
        assert_equal([[1, 2],
                      [3, 4]], t)

        # unstack(tensor, name=None): 纵向拆分, Tensor->TensorArray, 元素rank减1
        #   将tensor的第0维度拆开，每行作为TensorArray的一个元素。
        ta2 = tf.TensorArray(tf.int32, size=2)
        ta2.unstack(t)
        assert_equal_ta(ta1, ta2)

    def test_concat_split(self):
        # TensorArray <-> Tensor: 横向变换

        ta1 = tf.TensorArray(tf.int32, size=2)
        ta1.write(0, tf.constant([1, 2]))
        ta1.write(1, tf.constant([3, 4]))

        # concat(name=None): 横向堆叠，TensorArray->Tensor， rank保持
        assert_equal([1, 2, 3, 4], ta1.concat())

        # split(tensor, lengths=None): 横向拆分，Tensor->TensorArray, rank保持
        ta2 = tf.TensorArray(tf.int32, size=2)
        ta2.split([1, 2, 3, 4], lengths=[2, 2])
        assert_equal_ta(ta1, ta2)

        ta1 = tf.TensorArray(tf.int32, size=2)
        ta1.write(0, tf.constant([[1]]))
        ta1.write(1, tf.constant([[3]]))
        print(ta1.concat())
        print(ta1.stack())

    def test_gather_scatter(self):
        ## TensorArray <-> Tensor: 通过indices指定子集
        ta1 = tf.TensorArray(tf.int32, size=3)
        ta1.write(0, tf.constant([1, 2]))
        ta1.write(1, tf.constant([3, 4]))
        ta1.write(2, tf.constant([5, 6]))

        # gather(indices, name=None): TensorArray -> Tensor，默认是行堆叠
        t = ta1.gather(indices=[0, 2])
        assert_equal([[1, 2],
                      [5, 6]], t)

        # scatter(indices, tensor, name=None): Tensor -> TensorArray
        ta2 = tf.TensorArray(tf.int32, size=3)
        ta2.write(1, tf.constant([3, 4]))
        ta2.scatter([0, 2], tf.constant([[1, 2],
                                        [5, 6]]))
        assert_equal_ta(ta1, ta2)
