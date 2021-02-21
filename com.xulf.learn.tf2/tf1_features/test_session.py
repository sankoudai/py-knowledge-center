import tensorflow as tf
import tensorflow.compat.v1 as tf1
import unittest
from cmp_util import *
from except_util import *

class TestVersion(unittest.TestCase):
    def testSession(self):
        # tf2没有Session
        # tf1不同Session维护不同的变量
        w = tf1.Variable(10)

        sess1 = tf1.Session()
        sess2 = tf1.Session()

        assert_equal(20, sess1.run(w.assign_add(10)))
        assert_equal(30, sess1.run(w.assign_add(10)))

        assert_equal(20, sess2.run(w.assign_add(10)))
