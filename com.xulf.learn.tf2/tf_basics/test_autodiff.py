import tensorflow as tf
import unittest
from cmp_util import *
from except_util import *
from numpy.testing import assert_raises

class TestAutoDiff(unittest.TestCase):
    '''

        tf主要通过tf.GradientTape类实现求导
        1. 记录forward pass中的op及顺序:
        2. 在backward pass 逆序变量op序列，计算微分
        示例：
        ```
            x = tf.Variable(1.0)
            with tf.GradientTape() as tape: #记录开始
                y = x ** 2                  #记录结束
            grad = tape.gradient(y, x)      #求导
        ```

        参考:
        1. https://www.tensorflow.org/guide/autodiff
    '''

    def test_record(self):
        # tf.GradientTape(persistent=False, watch_accessed_variables=True)
        #   persistent:
        #       False: tape.gradient(target, sources) 执行一次就释放资源，二次调用报错
        #       True:  tape.gradient可执行多次，tape变量释放后，资源释放
        #   watch_accessed_variables:
        #       False: 不自动记录tape范围内的变量上的op, 需要主动watch感兴趣的变量
        #       True:  自动记录

        # 默认状态
        x = tf.Variable(1.0)
        with tf.GradientTape() as tape:
            y = x**2

        assert_equal(2.0, tape.gradient(y, x))
        with assert_raises(Exception):
            tape.gradient(y, x) #二次调用，报一次

        # persistent: True
        x = tf.Variable(1.0)
        with tf.GradientTape(persistent=True) as tape:
            y = x**2

        assert_equal(2.0, tape.gradient(y, x))
        assert_equal(2.0, tape.gradient(y, x))
        del tape

        # watch_accessed_variables: False
        x1 = tf.Variable(1.0)
        x2 = tf.Variable(2.0)
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(x1)
            y = x1 + 2*x2
        assert tape.gradient(y, x1) == 1.0
        assert tape.gradient(y, x2) is None

    def test_connect(self):
        # y and variable x are not connected,
        #   if x is not watched or x is not trainable variable

        # A trainable variable
        x0 = tf.Variable(3.0, name='x0')
        # Not trainable
        x1 = tf.Variable(3.0, name='x1', trainable=False)
        # Not a Variable: A variable + tensor returns a tensor.
        x2 = tf.Variable(2.0, name='x2') + 1.0
        # Not a variable
        x3 = tf.constant(3.0, name='x3')

        with tf.GradientTape() as tape:
          y = (x0**2) + (x1**2) + (x2**2)

        grads = tape.gradient(y, [x0, x1, x2, x3])
        assert_equal(grads, [6, None, None, None])

    def test_gradient(self):
        # tape.gradient
        #   方法签名：gradient(target, sources,..,unconnected_gradients=tf.UnconnectedGradients.NONE)
        #       target: 一般是scalar；如果是tensor，则先求和再求导
        #       sources: tensor
        #       unconnected_gradients: 如果target与source没有关系（或没被tape记录）， 返回的默认值

        # sources can be list or dict
        x1 = tf.Variable(1.0)
        x2 = tf.Variable(2.0)
        with tf.GradientTape(persistent=True) as tape:
            y = x1 + 2*x2

        grads = tape.gradient(y, [x1, x2])
        assert_equal([1, 2], grads)

        grads = tape.gradient(y, {'x1':x1, 'x2':x2})
        assert grads['x1'] == 1.0
        assert grads['x2'] == 2.0
        del tape

        # gradient with respect to model parameters:
        #   All subclasses of tf.Module aggregate their variables in tf.Module.trainable_variables
        #   So it can be done as follows:
        #       (tf.layers.Layer, tf.keras.Model are subclasses of tf.Module)
        layer = tf.keras.layers.Dense(2, activation='relu')
        x = tf.constant([[1., 2., 3.]])
        with tf.GradientTape() as tape:
          # Forward pass
          y = layer(x)
          loss = tf.reduce_mean(y**2)

        grads = tape.gradient(loss, layer.trainable_variables)
        print(grads)

