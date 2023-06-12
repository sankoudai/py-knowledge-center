import tensorflow as tf
import tensorflow.compat.v1 as tf1
import unittest
from cmp_util import *
from except_util import *

class TestVersion(unittest.TestCase):
    def testSession(self):

        colors_column = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(
            key='color',
            vocabulary_list=["Green", "Red", "Blue", "Yellow"]
        ))


        input_layer = tf.feature_column.input_layer(
            features={'color': tf.constant(value="Red", dtype=tf.string, shape=(1,))},
            feature_columns=[colors_column])


        with tf.Session() as sess:
            sess.run(tf.initialize_all_tables())
            print(sess.run(input_layer))

