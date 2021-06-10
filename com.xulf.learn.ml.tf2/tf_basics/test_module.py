import tensorflow as tf
from tensorflow import debugging as tfd
import tensorflow.compat.v1 as tf1
import unittest
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 为了兼容mac上的bug


class TestModule(unittest.TestCase):
    '''
        tf.Module(name=None)
        1. 定义： tf.Variable, tf.Module, functions(作用在input tensor)的named container.
                (tf.keras.layers.Layer, tf.keras.Model都是其子类)
        2. 核心功能：实例的tf.Variable、tf.Module类型properties，
                   会被自动收集到variables, trainable_variables和submodules属性上.

        2. attributes:
            - variables: sequences of variables owned by this module and its submodules
            - trainables: variables的可训练子集
            - submodules: 实例的tf.Module类型属性，及其tf.Module类型属性， ...
        3. methods:
            __call__： 一般module都会定义， 使其实例可当函数调用

        参考：
        1. https://www.tensorflow.org/api_docs/python/tf/Module
        2. https://www.tensorflow.org/guide/intro_to_modules#defining_models_and_layers_in_tensorflow
    '''

    def test_attributes(self):
        class Dense(tf.Module):
            def __init__(self, input_size, out_size, name=None):
                super(Dense, self).__init__(name=name)
                with self.name_scope:  # 用于变量分组， 便于区分与展示(如在tensorboard)
                    self.w = tf.Variable(tf.random.normal([input_size, out_size]), name='w')
                    self.b = tf.Variable(tf.random.normal([out_size]), name='b')
                    self.c = tf.Variable(1., name='c', trainable=False)

            def __call__(self, x):
                return x @ self.w + self.b

        class SeqModule(tf.Module):
            def __init__(self, name=None):
                super(SeqModule, self).__init__(name)
                self.dense_1 = Dense(3, 2)
                self.dense_2 = Dense(2, 2)

            def __call__(self, x):
                x = self.dense_1(x)
                x = self.dense_2(x)
                return x

        model = SeqModule(name='mlp')

        # variables
        assert model.variables == (model.dense_1.b, model.dense_1.c, model.dense_1.w,
                                   model.dense_2.b, model.dense_2.c, model.dense_2.w)

        # trainable_variables
        assert model.trainable_variables == (model.dense_1.b, model.dense_1.w,
                                             model.dense_2.b, model.dense_2.w)

        # submodules
        assert model.submodules == (model.dense_1, model.dense_2)

    def test_usecase(self):
        # 一般的tf.Module定义样子
        class Dense(tf.Module):
            def __init__(self, input_size, out_size, name=None):
                super(Dense, self).__init__(name=name)
                self.w = tf.Variable(tf.random.normal([input_size, out_size]), name='w')
                self.b = tf.Variable(tf.random.normal([out_size]), name='b')

            def __call__(self, x):
                return x @ self.w + self.b

        dense = Dense(3, 2)
        t = dense(tf.constant([[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0]]))
        tfd.assert_equal(t.shape, [2, 2])

        # 推迟tf.Variable的创建
        class FlexibleDense(tf.Module):
            def __init__(self, out_size, name=None):
                super(FlexibleDense, self).__init__(name=name)
                self.out_size = out_size
                self.built = False

            def __call__(self, x):
                if not self.built:
                    self.w = tf.Variable(tf.random.normal([x.shape[-1], self.out_size]))
                    self.b = tf.Variable(tf.random.normal([self.out_size]))
                return x @ self.w + self.b

<<<<<<< HEAD:com.xulf.learn.tf2/tf_basics/test_module.py
        flexible_dense = Dense(2)
        t = flexible_dense(tf.constant([[1.0, 2.0, 3.0],
=======
        flexible_dense = FlexibleDense(2)
        t = dense(tf.constant([[1.0, 2.0, 3.0],
>>>>>>> 566cf25349793a8941b1c27680794458e54b4be5:com.xulf.learn.ml.tf2/tf_basics/test_module.py
                               [4.0, 5.0, 6.0]]))
        tf.assert_equal(t.shape, [2, 2])
