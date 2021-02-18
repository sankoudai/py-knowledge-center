import tensorflow as tf
import tensorflow.compat.v1 as tf1
import unittest
from cmp_util import *
from except_util import *

class TestVariable(unittest.TestCase):
    '''
        参考： https://www.tensorflow.org/api_docs/python/tf/Variable
        tf.Variable: 可变tensor
            1. tf.Variable是一个类， 其实例有多种方法
            2. 其可作为一个普通tensor使用，其值可变
    '''

    def test_create(self):
        # 通过构造函数:
        #   方法签名：tf.Variable(initial_value=None, trainable=False, dtype=None, name=None, ..)
        #       initial_value: tensor 或可转化为tensor的对象（如python list, np array)
        v = tf.Variable([1.0], name='local_var')

    def test_update(self):
        # assign
        v = tf.Variable(2.0)

        v.assign(v*2)
        assert_equal(4.0, v)

        v.assign(v*2)
        assert_equal(8.0, v)

    def test_tf1_variable_init(self):
        tf1.disable_v2_behavior()

        w = tf1.get_variable('big_matrix', shape=(2, 2), initializer=tf.initializers.zeros())
        init_op = tf1.global_variables_initializer()
        with tf1.Session() as sess:
            sess.run(init_op)
            w_val = sess.run(w)
            assert_equal(np.zeros((2,2)), w_val)


    def test_tf1_variable_sharing(self):
        tf1.disable_v2_behavior()

        # 全局命名空间
        #   tf1 globals:  tf1存在implicit的全局命名空间， 使用tf.Variable创建的变量，都会记入默认的graph；
        #               即使程序已无引用， 变量仍会存在。 tf1的Variable可以通过name来共享， tf1程序的一个特点就是
        #               大量的全局变量共享(通过tf.get_variable创建的变量可共享）。 tf1 通过tf.variable_scope
        #               与 tf.get_variable实现变量共享。
        #   tf2 removes globals: tf2去除了全局命名空间， 只能通过函数参数的方式来传参， 鼓励函数式编程。 在tf2中
        #               tf.get_variable与tf.variable_scope都已经去除

        #  变量空间: tf1.variable_scope
        #      1. tf1的每个变量都处在（嵌套的)变量空间之内
        with tf1.variable_scope('a'):
            with tf1.variable_scope('b'):
                v = tf1.Variable(1.0, name='c')
                assert v.op.name == 'a/b/c'
                assert v.name=='a/b/c:0'

        # 同名变量，内部名不同
        #   说明：创建两个Variable， 如果name设置成一样， 则第二个名字会被修改
        v1 = tf1.Variable([1.0], name='my_var')
        v2  = tf1.Variable([1.0], name='my_var')
        assert v1.op.name == 'my_var'
        assert v2.op.name == 'my_var_1'


        # 通过tf.get_variable共享变量: 根据名字获取一个已存在的Variable，如果没有创建一个
        #   方法签名：tf1.get_variable(name, shape=None, dtype=None, initializer=None,
        #                   regularizer=None, trainable=None, collections=None)
        #   说明: variable_scope的reuse属性
        #       1. 如果False, 若已存在名字为name的变量，报错
        #       2. 如果True， 若不存在名字为name的变量，报错
        #       3. 如果tf.AUTO_REUSE, 若不存在则创建， 若已存在则复用，是使用最多的pattern.

        # reuse=False
        with tf1.variable_scope('a', reuse=False):
            W = tf1.get_variable('big_matrix', shape=(20, 20), initializer=tf.zeros_initializer())
            get_var = lambda : tf1.get_variable('big_matrix')
            assert_except(get_var)

        # reuse=True
        with tf1.variable_scope('b', reuse=True):
            get_var = lambda : tf1.get_variable('v', shape=(20, 20))
            assert_except(get_var)

        # reuse=tf.AUTO_REUSE
        with tf1.variable_scope('c', reuse=tf1.AUTO_REUSE):
            w1 = tf1.get_variable('big_matrix', shape=(20, 20), initializer=tf.zeros_initializer())
            w2 = tf1.get_variable('big_matrix')
            assert w1 is w2

        # 通过tf1.global_variables()共享变量
        w1 = tf1.Variable(1.0, name='foo')
        w2 = [v for v in tf1.global_variables() if v.op.name=='foo'][0]
        assert w1 is w2

        # 通过tf1.get_collection()共享变量
        tf1.reset_default_graph()
        w1 = tf1.Variable(1.0, name='foo')
        w2 = [v for v in tf1.get_collection(tf1.GraphKeys.GLOBAL_VARIABLES) if v.op.name=='foo'][0]
        assert w1 is w2

        # 创建自己的collection来共享变量
        tf1.reset_default_graph()
        w = tf1.Variable(1.0, name='foo')
        tf1.add_to_collection('my_collection', w)
        vars = tf1.get_collection('my_collection')
        assert w is vars[0]

    # def test_run(self):
