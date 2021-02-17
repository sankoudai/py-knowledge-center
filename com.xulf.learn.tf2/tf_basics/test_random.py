import tensorflow as tf
from cmp_util import *
from unittest import TestCase

class RandomTest(TestCase):
    """
    参考：
        API: https://www.tensorflow.org/api_docs/python/tf/random/categorical
        概率分布：https://www.statlect.com/probability-distributions/
    """

    def test_seed(self):
        # - tf中的随机操作由两个seeds决定：global seed 和 op seed
        #   global seed通过tf.random.set_seed设置, op seed一般通过op的参数设定
        #   tf1: tf.set_random_seed(seed)

        #  global seed 和 op seed 都不设置： 每次随机选取op种子
        t1 = tf.random.uniform([1])  # 每次运行程序都不同
        assert_ne(t1, 0.2180171)  # 0.2180171是某次运行结果

        # global seed 设置， op seed不设置： 系统确定性选择op seed (可能随tf版本、用户代码改动改变)
        tf.random.set_seed(1000)
        t1 = tf.random.uniform([1])
        t2 = tf.random.uniform([1])
        assert_equal(t1, 0.651229)
        assert_equal(t2, 0.6543844)

        #  同时设置global seed 和 op seed：得到确定性的序列结果
        tf.random.set_seed(1000)
        t1 = tf.random.uniform([1], seed=100)
        t2 = tf.random.uniform([1], seed=100)
        assert_equal(t1, 0.6579577)
        assert_equal(t2, 0.7809596)

        # op内部会保持一个计数， 所以连续调用返回不同的结果
        # tf.random.set_seed会重置这个计数器
        tf.random.set_seed(1000)
        t1 = tf.random.uniform([1], seed=100)
        tf.random.set_seed(1000)
        t2 = tf.random.uniform([1], seed=100)
        assert_equal(t1, 0.6579577)
        assert_equal(t2, 0.6579577)

        # 对tf2来说，@tf.function会使得op内部的计数器无法共享， 设置op的seed会导致结果一致
        @tf.function
        def foo():
            a = tf.random.uniform([1], seed=1)
            b = tf.random.uniform([1], seed=1)
            return a, b

        t1, t2 = foo()
        assert_equal(t1, t2)

    def test_distribution(self):
        # 均匀分布：
        # tf.random.uniform(shape, minval=0, maxval=None, dtype=tf.dtypes.float32, seed=None, name=None)
        # tf1: tf.random_uniform()
        # 说明：取值范围[minval, maxval)； dtype=float32时， 默认取值范围是[0, 1)
        tf.random.set_seed(100)
        t = tf.random.uniform([6], seed=100)
        assert_equal(t, [0.6834769, 0.628494, 0.55020416, 0.04058826, 0.28132665, 0.7678064], tol=1e-6)

        # 正态分布：
        # tf.random.normal(shape, mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, seed=None, name=None)
        # tf1: tf.random_normal
        tf.random.set_seed(100)
        t = tf.random.normal([6], mean=0, stddev=1.0, seed=100)
        assert_equal(t, [-0.6302907, -0.60320675, 0.27576178, 1.0577745, -1.582675, 0.17781347], tol=1e-6)

        # 截断正态分布：
        # tf.random.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, seed=None, name=None)
        # 说明：|x - mean| > 2*stddev 的取值被截断，并重新归一化.  相当于已知x∈[mean-2*stddev, mean+2*stddev]时的条件概率
        #      tensorflow_probality 子包里的对应分布可以设定x的取值范围
        mean, stddev = 1.0, 0.49
        t = tf.random.truncated_normal([100, 2], mean=mean, stddev=stddev)
        assert_greater(t, mean - 2 * stddev)
        assert_less(t, mean + 2 * stddev)

        # 多项分布：（方法设计不同于其他，我认为是设计败笔）
        # tf.random.categorical(logits, num_samples, dtype=None, seed=None, name=None)
        # tf1: tf.multinomial
        # 说明：
        #    logits:  2d tensor，[bath_size, num_class]；每一行logits[i,:]代表一个多项分布的对数概率
        #    num_samples: int,  每一个概率分布类， 随机采样数
        #    返回值为 [batch_size, num_samples]的2d tensor
        tf.random.set_seed(100)
        logits = tf.math.log([[0.5, 0.5, 0.0],  # 二项分布，取值0/1，概率均为0.5
                              [0.5, 0.4, 0.1]  # 3项分布, 取值0/1/2，概率分别为0.5, 0.4, 0.1
                              ])
        batch_size = logits.shape[0]
        num_samples = 5
        t = tf.random.categorical(logits, num_samples, seed=100)
        assert_shape(t, [batch_size, num_samples])
        assert_equal(t, [[0, 0, 0, 1, 0],
                         [0, 0, 1, 0, 2]])

    def test_stateless_distribution(self):
        # 分布基本都有一个无状态的版本
        # 与有状态不同的是， 给定seed，stateless版本的每次返回结果相同（在非cpu/gpu架构上可能不同）
        seed = [100, 100]
        t1 = tf.random.stateless_uniform([6], seed=seed)
        t2 = tf.random.stateless_uniform([6], seed=seed)
        assert_equal(t1, t2)

    def test_shuffle(self):
        # tf.random.shuffle(tensor, seed=None, name=None)
        # 作用：将一个tensor沿着第一维度（行维度）随机打散
        tf.random.set_seed(100)
        t = tf.constant([
            [1, 4],
            [2, 5],
            [3, 6]
        ])
        shuffled_t = tf.random.shuffle(t, seed=30)
        assert_equal(shuffled_t, [[2, 5],
                                  [1, 4],
                                  [3, 6]])

    def test_random_crop(self):
        # tf.image.random_crop(tensor, shape, seed=None, name=None)
        # 说明： 把一个tensor随机裁剪到shape
        #   shape: 1d tensor, 必须与tensor.shape具有相同的大小;
        #       如果某个维度不想裁剪， 则取tensor在该维度的大小， 如对RGB图片取[crop_height, crop_width, 3]

        tf.random.set_seed(100)
        t = tf.random.uniform([10, 10], seed=100)
        croped_t = tf.image.random_crop(t, [5, 5], seed=100)
        assert_shape(croped_t, [5, 5])
