import tensorflow as tf
import tensorflow.compat.v1 as tf1
from cmp_util import *
from except_util import *
import unittest

class TestFunction(unittest.TestCase):
    '''
        tf.function:
            参考：
            - RFC: https://github.com/tensorflow/community/pull/20/files
            - guide: https://www.tensorflow.org/guide/function
                     https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/control_flow.md

        说明：
        execution mode
            1. tf1 只有graph mode， 一个程序的执行分两步， 先构建graph；然后执行子图，执行的入口是sess.run(ops, feed_dict={})。
                sess.run每个op都定义了一个子图，op之间没有先后关系， 相当于分别执行。
            2. tf2 有graph mode 和 eagar mode两种， 程序默认按eagar执行。 eagar mode与graph mode的最大差别， 在于流程控制。
                eagar mode 中， 每个op会被立即执行并返回结果(concrete tensor)， 不会构建graph；所以运行tf与普通包无差别，执行流程
                完全是由python语言执行顺序确定，复杂的执行流程用eagar mode来写会非常明了。
                graph mode是先编译子图，然后执行； 在tf2中， 通过tf.function来实现与sess.run类似的功能。

        tf.function
            在概念上讲，tf.function(f) 与sess.run(op, feed_dict)是相同的， 本质都是执行一个子图。相比sess.run，tf.function带来的好处
            有2个：
            Pros:
                一是概念更清晰，一个tf.function对应一个子图的执行；
                二是op依赖关系指定更简单，当需要指定执行顺序时，在sess.run中需通过tf.control_dependencies(preceding_ops)来指定，
                    tf.function则直接通过python代码先后来指定执行顺序。 其底层机制是， 生产或消费tf.Variable （更一般的
                    DT_RESOURCE tensor)的ops，按照图生产顺序执行。
            Cons:
                tf.function也对程序引入了复杂性。
                 ```
                    @tf.function
                    def f(p1, p2):
                        pass
                 ```

                 调用f(p1, p2)时，  首先进行一次tracing，生产tf.Graph；然后tf.Graph会执行其中的tf op.
                1. tracing: 执行f中的python code, tf op会记入tf.Graph但不会执行。
                     !!!同样类型的参数tracing只进行一次！ 所以第二次调用时，不会重复执行其中的python code!
                     同样类型参数的定义：
                     a. tf.tensor - shape & type
                     b. python obj - 其id
                     所以调用时尽可能使用tf.tensor
                2. run: 执行tf.Graph中的ops

            坑：
            1. tf.function中只能第一次执行时，创建Variable
            2. tf.function中不要依赖可变python object (tf.Varaible可以）， 执行中感知不到
            3. tf.function尽可能简单，尽可能使用tf op
    '''

    def test_sess_vs_function1(self):
        # tf1: sess方式需要通过control_dependencies来指定incre_op与y+x的先后顺序
        tf1.disable_v2_behavior()
        y = tf1.Variable(1.0, dtype=tf1.float32)
        incre_op = tf1.assign_add(y, 10)

        x = tf1.placeholder(tf1.float32)
        with tf1.control_dependencies([incre_op]):
            y = tf1.add(y, x)

        init_op = tf1.global_variables_initializer()
        with tf1.Session() as sess:
            sess.run(init_op)
            val = sess.run(y, feed_dict={x:1})
            assert_equal(val, 12)

    def test_sess_vs_function2(self):
        # tf2: function方式
        @tf.function
        def incre_add(y, x):
            y = y + 10
            y = y + x
            return y

        y = tf.Variable(1.0, dtype=tf.float32)
        val = incre_add(y, tf.constant(1.0))
        assert_equal(val, 12)

    def test_tracing(self):
        xs = []
        @tf.function
        def f(x):
            xs.append(x)
            return tf.add(2, 1)

        # tensor of same shape and type:  tracing once
        xs = []
        f(tf.constant(1, dtype=tf.int32))
        assert len(xs) == 1
        f(tf.constant(2, dtype=tf.int32))
        assert len(xs) == 1

        # tensor of different shape or type: trace again, generate another graph
        f(tf.constant(1, dtype=tf.float32))
        assert len(xs) == 2

        # python primitive: or

    def test_use_patterns(self):
        # creating Variable in function:  should be done at first call
        v = None

        @tf.function
        def f_ok():
            nonlocal v
            if v is None:
                v = tf.Variable(1.0)
            return tf.add(1.0, v)
        assert_no_except(f_ok)

        @tf.function
        def f_fail():
            nonlocal v
            v = tf.Variable(1.0)
            return tf.add(1.0, v)
        assert_except(f_fail)

        # class method: self is python object, so each instance has different tracing
        class AModel(object):
            def __init__(self):
                self.v = None

            @tf.function
            def incre(self, amount):
                if self.v is None:
                    self.v = tf.Variable(tf.zeros_like(amount))
                self.v.assign_add(amount)
                return self.v

        model = AModel()
        assert_equal(1.0, model.incre(tf.constant(1.0)))
        assert_equal(2.0, model.incre(tf.constant(1.0)))
        model = AModel()
        assert_equal(1.0, model.incre(tf.constant(1.0)))
        assert_equal(2.0, model.incre(tf.constant(1.0)))

    def test_autograph(self):
        # Within tf.function, tf2 automatically transforms a subset of python eagar code to graph-mode ops.
        #   This includes if, for, while under certain conditions
        #   1. if <condition>: when condition is Tensor, convert to tf.cond
        #   2. for x in y: when y is tensor, convert to tf.while_loop
        #   3. while <condition>: when condition is tensor, convert to tf.while_loop
        #
        #  Difference between python-if and tf-if:
        #   1. python-if: tracing will include only one branch of if;
        #   2. tf-if: tf.cond will trace both branch, dynamically choosing branch in execution
        #
        #   Difference between python-loop and tf-loop:
        #   1. python-loop: add op to tf.Graph for every iteration during tracing. (when called twice, same ops)
        #   2. tf-loop: trace loop body and dynamically decide how many iterations to run in execution
        #
        #   Note: python values become tensors if modified in tf control flow

        # if
        i = []
        @tf.function
        def if_auto():
            if tf.greater(np.random.uniform(), 0.5):
                i.append(1)
            else:
                i.append(2)
            return tf.constant(1.0)
        if_auto()
        assert len(i) == 2
        assert_equal([1, 2], i)

        i = []
        @tf.function
        def if_not_auto():
            if np.random.uniform() > 0.5:
                i.append(1)
            else:
                i.append(2)
        if_not_auto()
        assert len(i) == 1

        # while
        i = []
        @tf.function
        def while_auto():
            t = 0.1
            while tf.less(t, 0.5):
                t += 0.1 # becomes a tensor
                i.append(1)
            return t

        t = while_auto()
        assert_equal(t, 0.5)
        assert t.dtype == tf.float32
        assert len(i)==1

        i = []
        @tf.function
        def while_not_auto():
            t = tf.constant(0)
            s = 0.0
            while s < 0.5:
                s += 0.1
                i.append(1)
                # tf ops
                t = tf.add(t, 1)
            return t

        t = while_not_auto()
        assert t == 5
        assert t.dtype == tf.int32
        assert len(i) == 5

        # for: accumulating data with TensorArray
        batch_size = 2
        seq_len = 3
        feature_size = 4

        def rnn_step(inp, state):
            return inp + state

        @tf.function
        def dynamic_rnn(rnn_step, input_data, initial_state):
            # [batch, time, features] -> [time, batch, features]
            input_data = tf.transpose(input_data, [1, 0, 2])
            max_seq_len = input_data.shape[0]

            states = tf.TensorArray(tf.float32, size=max_seq_len)
            state = initial_state
            for i in tf.range(max_seq_len):
                state = rnn_step(input_data[i], state)
                states = states.write(i, state)
            return tf.transpose(states.stack(), [1, 0, 2])

        dynamic_rnn(rnn_step,
                    tf.random.uniform([batch_size, seq_len, feature_size]),
                    tf.zeros([batch_size, feature_size]))