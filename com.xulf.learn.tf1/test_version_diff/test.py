import tensorflow as tf
import unittest

class Test(unittest.TestCase):
    def test(self):
        t1 = tf.constant([1, 1])
        t2 = tf.constant([1, 2])
        t = tf.div(t1,t2)
        with tf.Session() as sess:
            print(sess.run(t))
